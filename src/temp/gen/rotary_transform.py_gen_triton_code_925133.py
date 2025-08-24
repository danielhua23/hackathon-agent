
import torch
import triton
import triton.language as tl
from typing import Union, Optional


@triton.jit
def rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        x_ptr = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        o_ptr = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seq_len = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        x_ptr = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        o_ptr = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads
    else_seq_len = seqlen
    current_seqlen = tl.where(IS_VARLEN, seq_len, else_seq_len)

    if pid_m * BLOCK_M >= current_seqlen:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk_half = tl.arange(0, BLOCK_K)

    if not INTERLEAVED:
        cos_ptr = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        sin_ptr = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        mask_cos_sin = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half)
        cos = tl.load(cos_ptr, mask=mask_cos_sin, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr, mask=mask_cos_sin, other=0.0).to(tl.float32)

        left = x_ptr + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        right = x_ptr + (rm[:, None] * stride_x_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_x_headdim)
        mask_lr = (rm[:, None] < current_seqlen) & (rk_half[None, :] < rotary_dim_half)

        x0 = tl.load(left,  mask=mask_lr, other=0.0).to(tl.float32)
        x1 = tl.load(right, mask=mask_lr, other=0.0).to(tl.float32)

        if CONJUGATE:
            sin = -sin
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

        tl.store(
            o_ptr + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim),
            out0,
            mask=mask_lr,
        )
        tl.store(
            o_ptr + (rm[:, None] * stride_out_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_out_headdim),
            out1,
            mask=mask_lr,
        )
    else:
        rk = tl.arange(0, 2 * BLOCK_K)
        cos_ptr = COS + (rm_cs[:, None] * rotary_dim_half + (rk[None, :] // 2))
        sin_ptr = SIN + (rm_cs[:, None] * rotary_dim_half + (rk[None, :] // 2))
        mask_cos_sin = (rm_cs[:, None] < seqlen_ro) & (rk[None, :] < rotary_dim)
        cos = tl.load(cos_ptr, mask=mask_cos_sin, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr, mask=mask_cos_sin, other=0.0).to(tl.float32)

        idx = x_ptr + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        mask_idx = (rm[:, None] < current_seqlen) & (rk[None, :] < rotary_dim)
        x0 = tl.load(idx, mask=mask_idx, other=0.0).to(tl.float32)

        idx1 = x_ptr + (rm[:, None] * stride_x_seqlen + (rk[None, :] ^ 1) * stride_x_headdim)
        x1 = tl.load(idx1, mask=mask_idx, other=0.0).to(tl.float32)

        if CONJUGATE:
            sin = -sin
        out = tl.where(rk[None, :] % 2 == 0, x0 * cos - x1 * sin, x0 * sin + x1 * cos)

        tl.store(
            o_ptr + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim),
            out,
            mask=mask_idx,
        )
    # copy tail
    if rotary_dim < x.shape[-1]:  # headdim in signature is rotary_dim, actual head dim is X.shape[-1]
        actual_headdim = x.shape[-1]
    else:
        actual_headdim = rotary_dim
    if rotary_dim < actual_headdim:
        cols_tail = rotary_dim + tl.arange(0, BLOCK_K)
        mask_tail = (cols_tail < actual_headdim) & (rm[:, None] < current_seqlen)
        tail_src = x_ptr + (rm[:, None] * stride_x_seqlen + cols_tail[None, :] * stride_x_headdim)
        tail_val = tl.load(tail_src, mask=mask_tail)
        tail_dst = o_ptr + (rm[:, None] * stride_out_seqlen + cols_tail[None, :] * stride_out_headdim)
        tl.store(tail_dst, tail_val, mask=mask_tail)


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
        assert max_seqlen is not None, "max_seqlen is required when cu_seqlens provided"
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen

    seqlen_ro, rotary_half = cos.shape
    rotary_dim = rotary_half * 2
    assert rotary_dim <= headdim
    assert cos.dtype == sin.dtype == x.dtype
    cos, sin = cos.contiguous(), sin.contiguous()

    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = x if inplace else torch.empty_like(x)
    if rotary_dim < headdim and not inplace:
        if not is_varlen:
            output[..., rotary_dim:].copy_(x[..., rotary_dim:])  # shape (B, S, H, D)
        else:
            output[:, :, rotary_dim:].copy_(x[:, :, rotary_dim:])  # shape (ΣS, H, D)

    BLOCK_K = {32: 32, 64: 64, 128: 128, 256: 256}.get(
        triton.next_power_of_2(rotary_half), 256
    )
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)

    rotary_kernel[grid](
        output,
        x,
        cos,
        sin,
        cu_seqlens,
        seqlen_offsets,
        seqlen,
        nheads,
        rotary_dim,
        seqlen_ro,
        seqlen // 128,
        output.stride(0) if not is_varlen else 0,
        output.stride(-3),
        output.stride(-2),
        output.stride(-1),
        x.stride(0)           if not is_varlen else 0,
        x.stride(-3),
        x.stride(-2),
        x.stride(-1),
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
