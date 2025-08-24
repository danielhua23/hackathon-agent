
import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
    seqlen, nheads, rotary_dim, seqlen_ro, CACHE_KEY_SEQLEN,
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        x_base = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        out_base = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
        seqlen_i = seqlen
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen_i = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        x_base = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        out_base = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen_i:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rm_full_mask = rm < seqlen_i

    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rm_cs_mask = rm_cs < seqlen_ro

    if not INTERLEAVED:
        rk_half = tl.arange(0, BLOCK_K // 2)
        rk_mask = rk_half[None, :] < rotary_dim_half

        # Real part
        off_x_real = x_base + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
        x_real = tl.load(off_x_real, mask=rm_full_mask[:, None] & rk_mask, other=0.0).to(tl.float32)
        # Imag part
        off_x_imag = x_base + rm[:, None] * stride_x_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_x_headdim
        x_imag = tl.load(off_x_imag, mask=rm_full_mask[:, None] & rk_mask, other=0.0).to(tl.float32)

        off_cos = rm_cs[:, None] * (rotary_dim // 2) + rk_half[None, :]
        cos = tl.load(COS + off_cos, mask=rm_cs_mask[:, None] & rk_mask, other=1.0).to(tl.float32)
        sin_val = tl.load(SIN + off_cos, mask=rm_cs_mask[:, None] & rk_mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            sin_val = -sin_val

        o_real = x_real * cos - x_imag * sin_val
        o_imag = x_real * sin_val + x_imag * cos

        tl.store(out_base + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim,
                 o_real, mask=rm_full_mask[:, None] & rk_mask)
        tl.store(out_base + rm[:, None] * stride_out_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_out_headdim,
                 o_imag, mask=rm_full_mask[:, None] & rk_mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        mask_k = rk[None, :] < rotary_dim
        rk_half_idx = rk // 2
        mask_k_half = rk_half_idx[None, :] < rotary_dim_half

        off_x = x_base + rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim
        x_vals = tl.load(off_x, mask=rm_full_mask[:, None] & mask_k, other=0.0).to(tl.float32)
        x0 = tl.where(rk[None, :] % 2 == 0, x_vals, 0)
        x1 = tl.where(rk[None, :] % 2 == 1, x_vals, 0)

        off_cos_sin = rm_cs[:, None] * (rotary_dim // 2) + rk_half_idx[None, :]
        cos = tl.load(COS + off_cos_sin, mask=rm_cs_mask[:, None] & mask_k_half, other=1.0).to(tl.float32)
        sin_val = tl.load(SIN + off_cos_sin, mask=rm_cs_mask[:, None] & mask_k_half, other=0.0).to(tl.float32)
        if CONJUGATE:
            sin_val = -sin_val

        out_even = x0 * cos - x1 * sin_val
        out_odd = x0 * sin_val + x1 * cos

        out_vals = tl.where(rk[None, :] % 2 == 0, out_even, out_odd)
        tl.store(out_base + rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim,
                 out_vals, mask=rm_full_mask[:, None] & mask_k)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        seqlen = max_seqlen
        batch = cu_seqlens.shape[0] - 1

    seqlen_ro, rotary_dim_half = cos.shape
    assert sin.shape == cos.shape
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim

    seqlen_ro_needed = seqlen
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        seqlen_offsets = seqlen_offsets.to(torch.int32).contiguous()
        seqlen_ro_needed += seqlen_offsets.max().item()
    else:
        seqlen_ro_needed += seqlen_offsets
    assert seqlen_ro >= seqlen_ro_needed

    cos = cos.contiguous()
    sin = sin.contiguous()

    output = x if inplace else torch.empty_like(x)
    if not inplace and rotary_dim < headdim:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = 32 if rotary_dim <= 32 else \
              64 if rotary_dim <= 64 else \
              128 if rotary_dim <= 128 else 256
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)

    cu_seqlens_ptr = cu_seqlens.int().contiguous() if is_varlen else None

    rotary_kernel[grid](
        output, x, cos, sin, cu_seqlens_ptr, seqlen_offsets,
        seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128,
        output.stride(0) if not is_varlen else 0,
        output.stride(1),
        output.stride(2),
        output.stride(3),
        x.stride(0) if not is_varlen else 0,
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_K=BLOCK_K,
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
        IS_VARLEN=is_varlen,
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
        BLOCK_M=BLOCK_M,
    )
    return output


##################################################################################################################################################





import torch



def test_apply_rotary():

    results = {}

    

    # Test case 1: Basic test with fixed sequence length and no interleaving

    batch, seqlen, nheads, headdim = 2, 128, 4, 64

    rotary_dim = 32

    x = torch.randn(batch, seqlen, nheads, headdim, device='cuda')

    cos = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    sin = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    output = apply_rotary(x, cos, sin)

    results['test_case_1'] = output.shape



    # Test case 2: Variable length sequences with interleaving

    total_seqlen, nheads, headdim = 256, 4, 64

    batch = 3

    cu_seqlens = torch.tensor([0, 100, 200, 256], device='cuda')

    max_seqlen = 128

    rotary_dim = 32

    x = torch.randn(total_seqlen, nheads, headdim, device='cuda')

    cos = torch.randn(max_seqlen, rotary_dim // 2, device='cuda')

    sin = torch.randn(max_seqlen, rotary_dim // 2, device='cuda')

    output = apply_rotary(x, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=True)

    results['test_case_2'] = output.shape



    # Test case 3: Conjugate flag enabled

    batch, seqlen, nheads, headdim = 2, 128, 4, 64

    rotary_dim = 32

    x = torch.randn(batch, seqlen, nheads, headdim, device='cuda')

    cos = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    sin = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    output = apply_rotary(x, cos, sin, conjugate=True)

    results['test_case_3'] = output.shape



    # Test case 4: Inplace operation

    batch, seqlen, nheads, headdim = 2, 128, 4, 64

    rotary_dim = 32

    x = torch.randn(batch, seqlen, nheads, headdim, device='cuda')

    cos = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    sin = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    output = apply_rotary(x, cos, sin, inplace=True)

    results['test_case_4'] = output.shape



    return results



result_gold = test_apply_rotary()
