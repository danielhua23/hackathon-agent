
import torch
import triton
import triton.language as tl
from typing import Optional, Union


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
    stride_out_nheads,
    stride_out_seqlen,
    stride_out_headdim,
    stride_x_batch,
    stride_x_nheads,
    stride_x_seqlen,
    stride_x_headdim,
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

    if not IS_VARLEN:
        x_ptr = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        out_ptr = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
        cos_ptr = COS + pid_batch * seqlen_ro * (rotary_dim // 2)
        sin_ptr = SIN + pid_batch * seqlen_ro * (rotary_dim // 2)
        seqlen_i = seqlen
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen_i = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        x_ptr = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        out_ptr = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads
        cos_ptr = COS
        sin_ptr = SIN

    if pid_m * BLOCK_M >= seqlen_i:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    # Masks
    mask_m = rm < seqlen_i
    mask_k_half = rk_half < (rotary_dim // 2)

    if not INTERLEAVED:
        # Non-interleaved: contiguous real and imag parts
        x_real_offset = x_ptr + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
        x_imag_offset = x_real_offset + (rotary_dim // 2) * stride_x_headdim

        x_real = tl.load(x_real_offset, mask=mask_m[:, None] & mask_k_half[None, :], other=0.0).to(tl.float32)
        x_imag = tl.load(x_imag_offset, mask=mask_m[:, None] & mask_k_half[None, :], other=0.0).to(tl.float32)

        cos_offset = cos_ptr + rm_cs[:, None] * (rotary_dim // 2) + rk_half[None, :]
        sin_offset = sin_ptr + rm_cs[:, None] * (rotary_dim // 2) + rk_half[None, :]

        cos = tl.load(cos_offset, mask=(rm_cs[:, None] < seqlen_ro) & mask_k_half[None, :], other=1.0).to(tl.float32)
        sin_val = tl.load(sin_offset, mask=(rm_cs[:, None] < seqlen_ro) & mask_k_half[None, :], other=0.0).to(tl.float32)

        if CONJUGATE:
            sin_val = -sin_val

        o_real = x_real * cos - x_imag * sin_val
        o_imag = x_real * sin_val + x_imag * cos

        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim,
                 o_real, mask=mask_m[:, None] & mask_k_half[None, :])
        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + (rotary_dim // 2 + rk_half[None, :]) * stride_out_headdim,
                 o_imag, mask=mask_m[:, None] & mask_k_half[None, :])
    else:
        # Interleaved: even indices real, odd indices imag
        rk_even = rk * 2
        rk_odd = rk * 2 + 1
        rk_half = rk // 2

        mask_k_even = (rk_even < rotary_dim)
        mask_k_odd = (rk_odd < rotary_dim)
        mask_k_half_ready = rk_half < (rotary_dim // 2)

        cos_offset = cos_ptr + rm_cs[:, None] * (rotary_dim // 2) + rk_half[None, :]
        sin_offset = sin_ptr + rm_cs[:, None] * (rotary_dim // 2) + rk_half[None, :]

        cos_val = tl.load(cos_offset, mask=(rm_cs[:, None] < seqlen_ro) & mask_k_half_ready[None, :], other=1.0).to(tl.float32)
        sin_val = tl.load(sin_offset, mask=(rm_cs[:, None] < seqlen_ro) & mask_k_half_ready[None, :], other=0.0).to(tl.float32)

        if CONJUGATE:
            sin_val = -sin_val

        x_even_offset = x_ptr + rm[:, None] * stride_x_seqlen + rk_even[None, :] * stride_x_headdim
        x_odd_offset = x_ptr + rm[:, None] * stride_x_seqlen + rk_odd[None, :] * stride_x_headdim

        x_even = tl.load(x_even_offset, mask=mask_m[:, None] & mask_k_even[None, :], other=0.0).to(tl.float32)
        x_odd = tl.load(x_odd_offset, mask=mask_m[:, None] & mask_k_odd[None, :], other=0.0).to(tl.float32)

        grouped_even = x_even.reshape([-1, x_even.shape[1] // 2, 2])
        grouped_odd = x_odd.reshape([-1, x_odd.shape[1] // 2, 2])

        grouped_even_t = grouped_even[:, :, 0]
        grouped_odd_t = grouped_odd[:, :, 0]

        out_even = grouped_even_t * cos_val - grouped_odd_t * sin_val
        out_odd = grouped_even_t * sin_val + grouped_odd_t * cos_val

        out_even_unpacked = out_even.reshape(x_even.shape)
        out_odd_unpacked = out_odd.reshape(x_even.shape)

        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + rk_even[None, :] * stride_out_headdim,
                 out_even_unpacked, mask=mask_m[:, None] & mask_k_even[None, :])
        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + rk_odd[None, :] * stride_out_headdim,
                 out_odd_unpacked, mask=mask_m[:, None] & mask_k_odd[None, :])


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
    batch, seqlen, nheads, headdim = x.shape
    batch_ro, seqlen_ro, rotary_dim_half = cos.shape

    assert batch == batch_ro, f"batch mismatch: {batch} != {batch_ro}"
    assert sin.shape == cos.shape
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim, f"rotary_dim ({rotary_dim}) must be <= headdim ({headdim})"
    assert cos.dtype == sin.dtype == x.dtype, "All dtypes must match"
    assert not (cu_seqlens is not None and max_seqlen is None), "max_seqlen is required with cu_seqlens"

    seqlen_ro_needed = seqlen
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.to(torch.int32).contiguous()
        seqlen_ro_needed += seqlen_offsets.max().item()
    else:
        seqlen_ro_needed += seqlen_offsets
    assert seqlen_ro >= seqlen_ro_needed, f"seqlen_ro ({seqlen_ro}) must be >= seqlen_ro_needed ({seqlen_ro_needed})"

    cos = cos.contiguous()
    sin = sin.contiguous()

    output = x if inplace else torch.empty_like(x)
    if not inplace and rotary_dim < headdim:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = 32 if rotary_dim <= 32 else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)

    IS_VARLEN = (cu_seqlens is not None)
    CU_SEQLENS_ptr = (cu_seqlens.int().contiguous() if IS_VARLEN else None)

    rotary_kernel[grid](
        output,
        x,
        cos,
        sin,
        CU_SEQLENS_ptr,
        seqlen_offsets,
        seqlen,
        nheads,
        rotary_dim,
        seqlen_ro,
        seqlen // 128,
        output.stride(0),
        output.stride(2),
        output.stride(1),
        output.stride(3),
        x.stride(0),
        x.stride(2),
        x.stride(1),
        x.stride(3),
        BLOCK_K=BLOCK_K,
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
        IS_VARLEN=IS_VARLEN,
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
