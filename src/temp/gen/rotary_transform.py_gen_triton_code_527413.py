
import torch
import triton
import triton.language as tl
import math
from typing import Optional, Union

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
        x_ptr = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        out_ptr = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        x_ptr = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        out_ptr = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < seqlen
    rm_cs = rm
    if IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    else:
        rm_cs = rm + SEQLEN_OFFSETS

    if not INTERLEAVED:
        rk_half = tl.arange(0, BLOCK_K)
        mask_k = rk_half < rotary_dim_half
        x0 = tl.load(x_ptr + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim,
                     mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + rm[:, None] * stride_x_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_x_headdim,
                     mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        cos = tl.load(COS + rm_cs[:, None] * rotary_dim_half + rk_half[None, :],
                      mask=(rm_cs[:, None] < seqlen_ro) & mask_k[None, :], other=1.0).to(tl.float32)
        sin = tl.load(SIN + rm_cs[:, None] * rotary_dim_half + rk_half[None, :],
                      mask=(rm_cs[:, None] < seqlen_ro) & mask_k[None, :], other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim,
                 o0, mask=mask_m[:, None] & mask_k[None, :])
        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_out_headdim,
                 o1, mask=mask_m[:, None] & mask_k[None, :])
    else:
        rk = tl.arange(0, BLOCK_K)
        mask_k = rk < rotary_dim
        x = tl.load(x_ptr + rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        rk_half = rk // 2
        mask_k_half = rk_half < rotary_dim_half
        cos = tl.load(COS + rm_cs[:, None] * rotary_dim_half + rk_half[None, :],
                      mask=(rm_cs[:, None] < seqlen_ro) & mask_k_half[None, :], other=1.0).to(tl.float32)
        sin = tl.load(SIN + rm_cs[:, None] * rotary_dim_half + rk_half[None, :],
                      mask=(rm_cs[:, None] < seqlen_ro) & mask_k_half[None, :], other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        cos = tl.where(rk[None, :] % 2 == 0, cos, cos)
        sin = tl.where(rk[None, :] % 2 == 0, sin, sin)
        x0 = x
        x1 = tl.roll(x, shifts=1, axis=1)
        x1 = tl.where(rk[None, :] % 2 == 0, x1, -x1)
        out = x0 * cos - x1 * sin
        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim,
                 out, mask=mask_m[:, None] & mask_k[None, :])


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
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"
    assert cos.dtype == sin.dtype, f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert x.dtype == cos.dtype, f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos = cos.contiguous()
    sin = sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro
        seqlen_offsets = int(seqlen_offsets)

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K_m = (
        32 if rotary_dim_half <= 32
        else 64 if rotary_dim_half <= 64
        else 128 if rotary_dim_half <= 128
        else 256
    )
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)

    rotary_kernel[grid](
        output, x, cos, sin, cu_seqlens, seqlen_offsets,
        seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128,
        output.stride(0) if not is_varlen else 0,
        output.stride(-3) if x.dim() == 4 else output.stride(-2),
        output.stride(-2) if x.dim() == 4 else output.stride(-1),
        output.stride(-1) if x.dim() == 4 else 1,
        x.stride(0) if not is_varlen else 0,
        x.stride(-3) if x.dim() == 4 else x.stride(-2),
        x.stride(-2) if x.dim() == 4 else x.stride(-1),
        x.stride(-1) if x.dim() == 4 else 1,
        BLOCK_K_m, isinstance(seqlen_offsets, torch.Tensor),
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
