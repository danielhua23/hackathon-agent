
import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def rotary_kernel(
    OUT,        # *float16/bfloat16/float32
    X,          # *float16/bfloat16/float32
    COS,        # *float16/bfloat16/float32
    SIN,        # *float16/bfloat16/float32
    CU_SEQLENS, # *int32
    SEQLEN_OFFSETS,    # *int32 when IS_SEQLEN_OFFSETS_TENSOR==True, otherwise scalar int
    seqlen,            # int32
    nheads,            # int32
    rotary_dim,        # int32
    seqlen_ro,         # int32
    CACHE_KEY_SEQLEN,  # int32 (unused in kernel; kept for signature match)
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    BLOCK_K    : tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR : tl.constexpr,
    IS_VARLEN  : tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE  : tl.constexpr,
    BLOCK_M    : tl.constexpr,
):
    pid_m    = tl.program_id(axis=0)
    pid_batch= tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    # Determine batch / seqlen per sample
    if IS_VARLEN == 0:
        # fixed-seqlen; X, OUT already point at or part of contiguous
        offset_b  = pid_batch * stride_x_batch
        offset_bo = pid_batch * stride_out_batch
        current_seqlen = seqlen
    else:
        seqlen_start = tl.load(CU_SEQLENS + pid_batch).to(tl.int32)
        seqlen_end   = tl.load(CU_SEQLENS + pid_batch + 1).to(tl.int32)
        current_seqlen = seqlen_end - seqlen_start
        offset_b  = seqlen_start * stride_x_seqlen
        offset_bo = seqlen_start * stride_out_seqlen

    # Compute linears
    X   += offset_b  + pid_head * stride_x_nheads
    OUT += offset_bo + pid_head * stride_out_nheads

    # Return early for empty/tail blocks
    if pid_m * BLOCK_M >= current_seqlen:
        return

    # Row indices and validity mask
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < current_seqlen

    # sequence length offset
    if IS_SEQLEN_OFFSETS_TENSOR:
        seq_offset = tl.load(SEQLEN_OFFSETS + pid_batch).to(tl.int32)
    else:
        seq_offset = SEQLEN_OFFSETS  # scalar integer captured at launch & constant in kernel

    # half-size dimension indices
    rotary_dim_half = rotary_dim // 2
    rk_half = tl.arange(0, rotary_dim_half)
    mask_half = rk_half < rotary_dim_half

    if INTERLEAVED == 0:
        # Non-interleaved layout  -------------------------------------------------
        base_pos = (rm[:, None] + seq_offset) * rotary_dim + rk_half[None, :]
        cos_mask = ((rm[:, None] + seq_offset) < seqlen_ro) & mask_half[None, :]
        sin_mask = cos_mask

        cos = tl.load(COS + base_pos, mask=cos_mask, other=1.0).to(tl.float32)
        sin = tl.load(SIN + base_pos, mask=sin_mask, other=0.0).to(tl.float32)

        x0_ptr = X   + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
        x1_ptr = X   + rm[:, None] * stride_x_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_x_headdim
        x0 = tl.load(x0_ptr, mask=mask_m[:, None] & mask_half[None, :], other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr, mask=mask_m[:, None] & mask_half[None, :], other=0.0).to(tl.float32)

        if CONJUGATE:
            sin = -sin

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        tl.store(OUT + x0_ptr - X + OUT, y0,
                 mask=mask_m[:, None] & mask_half[None, :])
        tl.store(OUT + x1_ptr - X + OUT, y1,
                 mask=mask_m[:, None] & mask_half[None, :])

        # remainder pass-through
        if rotary_dim < stride_x_headdim * stride_x_headdim or True:
            rk_rem = tl.arange(rotary_dim, stride_x_headdim)
            out_off = OUT + rm[:, None] * stride_out_seqlen + rk_rem[None, :] * stride_out_headdim
            x_off   = X   + rm[:, None] * stride_x_seqlen + rk_rem[None, :] * stride_x_headdim
            mask_rem = (rk_rem[None, :] < stride_x_headdim) & mask_m[:, None]
            val_rem = tl.load(x_off, mask=mask_rem, other=0.0)
            tl.store(out_off, val_rem, mask=mask_rem)

    else:
        # Interleaved layout  ----------------------------------------------------
        full_dim = rotary_dim
        rk = tl.arange(0, full_dim)
        mask_k = rk < full_dim
        rk_half_idx = rk // 2

        base_pos = (rm[:, None] + seq_offset) * full_dim + rk_half_idx[None, :]
        mask_pos = ((rm[:, None] + seq_offset) < seqlen_ro) & mask_k[None, :]

        cos_val = tl.load(COS + base_pos, mask=mask_pos, other=1.0).to(tl.float32)
        sin_val = tl.load(SIN + base_pos, mask=mask_pos, other=0.0).to(tl.float32)

        x_off = X + rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim
        x_val = tl.load(x_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        if CONJUGATE:
            sin_val = -sin_val

        # flip sin when odd indices
        sin_flipped = tl.where((rk[None, :] % 2) == 0, sin_val, -sin_val)
        out_val = x_val * cos_val + sin_flipped * x_val.roll(-1, axis=1)

        tl.store(OUT + rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim,
                 out_val, mask=mask_m[:, None] & mask_k[None, :])


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
    """Top-level wrapper for RoPE Triton kernel (AMD ROCm)."""
    if cu_seqlens is None:
        batch, seqlen, nheads, headdim = x.shape
        total_seqlen = batch * seqlen
        stride_batch = x.stride(0)
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.numel() - 1
        seqlen = max_seqlen
        stride_batch = 0  # unused in varlen mode

    seqlen_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim
    assert seqlen_ro >= seqlen
    assert cos.dtype == sin.dtype == x.dtype
    assert rotary_dim % 2 == 0

    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.dtype == torch.int32
        seqlen_offsets = seqlen_offsets.contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = triton.next_power_of_2(rotary_dim)

    grid_m = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads)
    BLOCK_M = 4 if interleaved else 8

    rotary_kernel[grid_m](
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
        0,          # CACHE_KEY_SEQLEN (placeholder, unused)
        *output.stride(),
        *x.stride(),
        BLOCK_K,
        isinstance(seqlen_offsets, torch.Tensor),
        cu_seqlens is not None,
        interleaved,
        conjugate,
        BLOCK_M,
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
