
import torch
import triton
import triton.language as tl
import math
from typing import Optional, Union

@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
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

    cu_seqlens_ptr = CU_SEQLENS
    seqlen_offsets_ptr = SEQLEN_OFFSETS

    if not IS_VARLEN or CU_SEQLENS is None:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
        cur_seqlen = seqlen
    else:
        seq_start = tl.load(cu_seqlens_ptr + pid_batch)
        cur_seqlen = tl.load(cu_seqlens_ptr + pid_batch + 1) - seq_start
        X = X + seq_start * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + seq_start * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= cur_seqlen:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(seqlen_offsets_ptr + pid_batch)

    if not INTERLEAVED:
        x0_ptr = X + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
        x1_ptr = X + rm[:, None] * stride_x_seqlen + (rk_half + rotary_dim_half)[None, :] * stride_x_headdim

        c_ptr = COS + rm_cs[:, None] * stride_sin_seqlen + rk_half[None, :] * stride_sin_headdim
        s_ptr = SIN + rm_cs[:, None] * stride_sin_seqlen + rk_half[None, :] * stride_sin_headdim

        mask_m = rm[:, None] < cur_seqlen
        mask_ro_k = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half)
        mask_x_k = mask_m & (rk_half[None, :] < rotary_dim_half)

        c = tl.load(c_ptr, mask=mask_ro_k, other=1.0).to(tl.float32)
        s = tl.load(s_ptr, mask=mask_ro_k, other=0.0).to(tl.float32)
        x0 = tl.load(x0_ptr, mask=mask_x_k, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr, mask=mask_x_k, other=0.0).to(tl.float32)

        if CONJUGATE:
            s = -s

        o0 = x0 * c - x1 * s
        o1 = x0 * s + x1 * c

        out0_ptr = OUT + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim
        out1_ptr = OUT + rm[:, None] * stride_out_seqlen + (rk_half + rotary_dim_half)[None, :] * stride_out_headdim

        tl.store(out0_ptr, o0, mask=mask_x_k)
        tl.store(out1_ptr, o1, mask=mask_x_k)
    else:
        offs_d = 2 * tl.arange(0, rotary_dim_half)
        x0_ptr = X + rm[:, None] * stride_x_seqlen + offs_d[None, :] * stride_x_headdim
        x1_ptr = X + rm[:, None] * stride_x_seqlen + (offs_d + 1)[None, :] * stride_x_headdim

        c_ptr = COS + rm_cs[:, None] * stride_sin_seqlen + tl.arange(0, rotary_dim_half)[None, :] * stride_sin_headdim
        s_ptr = SIN + rm_cs[:, None] * stride_sin_seqlen + tl.arange(0, rotary_dim_half)[None, :] * stride_sin_headdim

        mask_m = rm[:, None] < cur_seqlen
        mask_ro_k = (rm_cs[:, None] < seqlen_ro) & (tl.arange(0, rotary_dim_half)[None, :] < rotary_dim // 2)
        mask_x_k = mask_m & (tl.arange(0, rotary_dim_half)[None, :] < rotary_dim // 2)

        c = tl.load(c_ptr, mask=mask_ro_k, other=1.0).to(tl.float32)
        s = tl.load(s_ptr, mask=mask_ro_k, other=0.0).to(tl.float32)
        x0 = tl.load(x0_ptr, mask=mask_x_k, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr, mask=mask_x_k, other=0.0).to(tl.float32)

        if CONJUGATE:
            s = -s

        o0 = x0 * c - x1 * s
        o1 = x0 * s + x1 * c

        out0_ptr = OUT + rm[:, None] * stride_out_seqlen + offs_d[None, :] * stride_out_headdim
        out1_ptr = OUT + rm[:, None] * stride_out_seqlen + (offs_d + 1)[None, :] * stride_out_headdim

        tl.store(out0_ptr, o0, mask=mask_x_k)
        tl.store(out1_ptr, o1, mask=mask_x_k)

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
    """
    Apply rotary embedding to the input tensor x using Triton kernels optimized for AMD GPU ROCm.

    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
           else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int. Required if cu_seqlens is not None.
        interleaved: Use interleaved layout (rotary_dim = headdim // 2)
        inplace: Whether to perform the rotation in-place (x will be modified)
        conjugate: Whether to negate the sine component
    Returns:
        y: (batch, seqlen, nheads, headdim) or (total_seqlen, nheads, headdim) same shape as x
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        if max_seqlen is None:
            raise ValueError("max_seqlen must be provided if cu_seqlens is used")
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen

    seqlen_ro, rotary_dimhalf = cos.shape
    rotary_dim = rotary_dimhalf * 2
    assert sin.shape == cos.shape
    assert rotary_dim <= headdim, f"Rotary dimension={rotary_dim} must be <= head_dim={headdim}"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, f"seqlen_ro={seqlen_ro} must >= seqlen={seqlen}"
    assert cos.dtype == sin.dtype
    assert x.dtype == cos.dtype

    cos = cos.contiguous()
    sin = sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        if not is_varlen:
            output[..., rotary_dim:].copy_(x[..., rotary_dim:])
        else:
            output[:, :, rotary_dim:].copy_(x[:, :, rotary_dim:])

    BLOCK_K = (
        32 if rotary_dim <= 32 else
        64 if rotary_dim <= 64 else
        128 if rotary_dim <= 128 else 256
    )
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)

    # Set strides correctly depending on tensor shape
    if x.dim() == 4:
        x_stride_b, x_stride_seqlen, x_stride_h, x_stride_d = (
            x.stride(0), x.stride(1), x.stride(2), x.stride(3)
        )
        output_stride_b, output_stride_seqlen, output_stride_h, output_stride_d = (
            output.stride(0), output.stride(1), output.stride(2), output.stride(3)
        )
    else:  # x.dim() == 3
        x_stride_b, x_stride_seqlen, x_stride_h, x_stride_d = (
            0, x.stride(0), x.stride(1), x.stride(2)
        )
        output_stride_b, output_stride_seqlen, output_stride_h, output_stride_d = (
            0, output.stride(0), output.stride(1), output.stride(2)
        )

    cos_stride_m = cos.stride(0)
    cos_stride_n = cos.stride(1)
    sin_stride_m = sin.stride(0)
    sin_stride_n = sin.stride(1)

    global stride_sin_seqlen, stride_sin_headdim
    stride_sin_seqlen = cos_stride_m
    stride_sin_headdim = cos_stride_n

    with torch.cuda.device(x.device.index):
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
            seqlen // 128,  # cache key
            output_stride_b,
            output_stride_seqlen,
            output_stride_h,
            output_stride_d,
            x_stride_b,
            x_stride_seqlen,
            x_stride_h,
            x_stride_d,
            BLOCK_K=BLOCK_K,
            IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
            IS_VARLEN=is_varlen,
            INTERLEAVED=interleaved,
            CONJUGATE=conjugate,
            BLOCK_M=BLOCK_M
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
