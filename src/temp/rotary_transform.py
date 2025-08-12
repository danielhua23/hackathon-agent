
import torch
import triton
import triton.language as tl
from typing import Union, Optional

@triton.jit
def rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
    seqlen, nheads, rotary_dim, seqlen_ro, CACHE_KEY_SEQLEN,
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    BLOCK_K: tl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr, INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    if not INTERLEAVED:
        rk_half = tl.arange(0, BLOCK_K // 2)
        mask_h = rk_half[None, :] < rotary_dim_half

        x0 = tl.load(
            X + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & mask_h,
            other=0.0,
        ).to(tl.float32)

        x1 = tl.load(
            X + rm[:, None] * stride_x_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & mask_h,
            other=0.0,
        ).to(tl.float32)

        cos_val = tl.load(
            COS + rm_cs[:, None] * rotary_dim_half + rk_half[None, :],
            mask=(rm_cs[:, None] < seqlen_ro) & mask_h,
            other=1.0,
        ).to(tl.float32)

        sin_val = tl.load(
            SIN + rm_cs[:, None] * rotary_dim_half + rk_half[None, :],
            mask=(rm_cs[:, None] < seqlen_ro) & mask_h,
            other=0.0,
        ).to(tl.float32)

        if CONJUGATE:
            sin_val = -sin_val

        o0 = x0 * cos_val - x1 * sin_val
        o1 = x0 * sin_val + x1 * cos_val

        tl.store(
            OUT + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim,
            o0,
            mask=(rm[:, None] < seqlen) & mask_h,
        )
        tl.store(
            OUT + rm[:, None] * stride_out_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & mask_h,
        )
    else:
        rk_half = tl.arange(0, BLOCK_K // 2)
        rk = tl.arange(0, rotary_dim)
        mask_full = rk[None, :] < rotary_dim
        rk_even = rk % 2 == 0
        rk_arr = rk[None, :]
        rm_arr = rm[:, None]

        x0 = tl.load(
            X + rm_arr * stride_x_seqlen + rk_arr * stride_x_headdim,
            mask=(rm_arr < seqlen) & mask_full,
            other=0.0,
        ).to(tl.float32)

        rk_swap = rk + ((((rk + 1) % 2) * 2) - 1)
        x1 = tl.load(
            X + rm_arr * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim,
            mask=(rm_arr < seqlen) & mask_full,
            other=0.0,
        ).to(tl.float32)

        idx = rk // 2
        cos_val = tl.load(
            COS + rm_cs[:, None] * rotary_dim_half + idx[None, :],
            mask=(rm_cs[:, None] < seqlen_ro) & mask_full,
            other=1.0,
        ).to(tl.float32)
        sin_val = tl.load(
            SIN + rm_cs[:, None] * rotary_dim_half + idx[None, :],
            mask=(rm_cs[:, None] < seqlen_ro) & mask_full,
            other=0.0,
        ).to(tl.float32)

        if CONJUGATE:
            sin_val = -sin_val
        out = tl.where(rk_even[None, :], x0 * cos_val - x1 * sin_val, x0 * sin_val + x1 * cos_val)
        tl.store(
            OUT + rm_arr * stride_out_seqlen + rk_arr * stride_out_headdim,
            out,
            mask=(rm_arr < seqlen) & mask_full,
        )


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
        stride_batch, stride_seqlen, stride_heads, stride_headdim = x.stride(0), x.stride(1), x.stride(2), x.stride(3)
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        stride_batch, stride_seqlen, stride_heads, stride_headdim = x.stride(0), x.stride(1), x.stride(2), x.stride(3)
        batch = cu_seqlens.size(0) - 1
        seqlen = max_seqlen

    seqlen_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2
    assert sin.shape == cos.shape
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen

    assert cos.dtype == sin.dtype
    assert x.dtype == cos.dtype

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        if not is_varlen:
            output[..., rotary_dim:].copy_(x[..., rotary_dim:])
        else:
            cu_x = x.view(-1, headdim)
            cu_out = output.view(-1, headdim)
            cu_out[:, rotary_dim:].copy_(cu_x[:, rotary_dim:])

    rdim = rotary_dim
    BLOCK_K = 16 if rdim <= 32 else (32 if rdim <= 64 else (64 if rdim <= 128 else 128))
    BLOCK_M = 4 if interleaved else (8 if rdim <= 64 else 4)
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)

    rotary_kernel[grid](
        output, x, cos, sin, cu_seqlens, seqlen_offsets,
        seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128,
        stride_batch, stride_seqlen, stride_heads, stride_headdim,
        stride_batch, stride_seqlen, stride_heads, stride_headdim,
        BLOCK_K, isinstance(seqlen_offsets, torch.Tensor),
        is_varlen, interleaved, conjugate, BLOCK_M,
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
