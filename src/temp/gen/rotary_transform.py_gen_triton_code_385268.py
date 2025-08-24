
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
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    stride_cos_seqlen,
    stride_cos_dim,
    stride_sin_seqlen,
    stride_sin_dim,
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROTARY_DIM_HALF: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    if not IS_VARLEN:
        cur_seqlen = seqlen
        x_ptr = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        out_ptr = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        seq_start = tl.load(CU_SEQLENS + pid_batch)
        cur_seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - seq_start
        x_ptr = X + seq_start * stride_x_seqlen + pid_head * stride_x_nheads
        out_ptr = OUT + seq_start * stride_out_seqlen + pid_head * stride_out_nheads
    if pid_m * BLOCK_M >= cur_seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, BLOCK_K // 2)
    if IS_SEQLEN_OFFSETS_TENSOR:
        offset = tl.load(SEQLEN_OFFSETS + pid_batch)
    else:
        offset = SEQLEN_OFFSETS
    rm_cs = rm + offset
    rm_cs = tl.where(rm_cs < seqlen_ro, rm_cs, seqlen_ro - 1)
    if not INTERLEAVED:
        x0_ptr = x_ptr + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
        x1_ptr = x_ptr + rm[:, None] * stride_x_seqlen + (rk_half + ROTARY_DIM_HALF)[None, :] * stride_x_headdim
        cos_ptr = COS + rm_cs[:, None] * stride_cos_seqlen + rk_half[None, :] * stride_cos_dim
        sin_ptr = SIN + rm_cs[:, None] * stride_sin_seqlen + rk_half[None, :] * stride_sin_dim
        mask_m = rm[:, None] < cur_seqlen
        mask_k_half = rk_half[None, :] < ROTARY_DIM_HALF
        cos = tl.load(cos_ptr, mask=(rm_cs[:, None] < seqlen_ro) & mask_k_half, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr, mask=(rm_cs[:, None] < seqlen_ro) & mask_k_half, other=0.0).to(tl.float32)
        x0 = tl.load(x0_ptr, mask=mask_m & mask_k_half, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr, mask=mask_m & mask_k_half, other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim,
                 o0, mask=mask_m & mask_k_half)
        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + (rk_half + ROTARY_DIM_HALF)[None, :] * stride_out_headdim,
                 o1, mask=mask_m & mask_k_half)
    else:
        rk_even = 2 * tl.arange(0, ROTARY_DIM_HALF)
        rk_odd = 2 * tl.arange(0, ROTARY_DIM_HALF) + 1
        x0_ptr = x_ptr + rm[:, None] * stride_x_seqlen + rk_even[None, :] * stride_x_headdim
        x1_ptr = x_ptr + rm[:, None] * stride_x_seqlen + rk_odd[None, :] * stride_x_headdim
        cos_ptr = COS + rm_cs[:, None] * stride_cos_seqlen + tl.arange(0, ROTARY_DIM_HALF)[None, :] * stride_cos_dim
        sin_ptr = SIN + rm_cs[:, None] * stride_sin_seqlen + tl.arange(0, ROTARY_DIM_HALF)[None, :] * stride_sin_dim
        mask_m = rm[:, None] < cur_seqlen
        mask_half = tl.arange(0, ROTARY_DIM_HALF)[None, :] < ROTARY_DIM_HALF
        cos = tl.load(cos_ptr, mask=(rm_cs[:, None] < seqlen_ro) & mask_half, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr, mask=(rm_cs[:, None] < seqlen_ro) & mask_half, other=0.0).to(tl.float32)
        x0 = tl.load(x0_ptr, mask=mask_m & mask_half, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr, mask=mask_m & mask_half, other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + rk_even[None, :] * stride_out_headdim,
                 o0, mask=mask_m & mask_half)
        tl.store(out_ptr + rm[:, None] * stride_out_seqlen + rk_odd[None, :] * stride_out_headdim,
                 o1, mask=mask_m & mask_half)

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
    """Apply rotary embedding to the input tensor x using Triton kernels optimized for AMD GPU ROCm."""
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        if max_seqlen is None:
            raise ValueError("max_seqlen must be provided if cu_seqlens is used")
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen
    assert cos.dtype == sin.dtype == x.dtype

    cos = cos.contiguous()
    sin = sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32 if rotary_dim <= 32 else
        64 if rotary_dim <= 64 else
        128 if rotary_dim <= 128 else 256
    )
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)

    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output, x, cos, sin, cu_seqlens, seqlen_offsets,
            seqlen, nheads, rotary_dim, seqlen_ro,
            0,
            output.stride(0) if not is_varlen else 0,
            output.stride(-3), output.stride(-2), output.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3), x.stride(-2), x.stride(-1),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            BLOCK_K=BLOCK_K,
            IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
            IS_VARLEN=is_varlen,
            INTERLEAVED=interleaved,
            CONJUGATE=conjugate,
            BLOCK_M=BLOCK_M,
            ROTARY_DIM_HALF=rotary_dim_half
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
