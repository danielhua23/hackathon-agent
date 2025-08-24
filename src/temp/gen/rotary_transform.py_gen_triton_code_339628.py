
import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def rotary_kernel(
    OUT,  # *float32
    X,  # *float32
    COS,  # *float32
    SIN,  # *float32
    CU_SEQLENS,  # *int32
    SEQLEN_OFFSETS,  # *int32
    seqlen,  # int32
    rotary_dim,  # int32                 # rotary dimension (must be even)
    seqlen_ro,  # int32                # rotary sequence length
    stride_out_batch,  # int64
    stride_out_seqlen,  # int64
    stride_out_nheads,  # int64
    stride_out_headdim,  # int64
    stride_x_batch,  # int64
    stride_x_seqlen,  # int64
    stride_x_nheads,  # int64
    stride_x_headdim,  # int64
    BLOCK_K: tl.constexpr,  # rotary dimension (must be even)
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,  # bool
    IS_VARLEN: tl.constexpr,  # bool
    INTERLEAVED: tl.constexpr,  # bool
    CONJUGATE: tl.constexpr,  # bool
    BLOCK_M: tl.constexpr,  # block size along sequence dimension
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    if not IS_VARLEN:
        offset_b = pid_batch * stride_x_batch
        offset_bo = pid_batch * stride_out_batch
        current_seqlen = seqlen
    else:
        seqlen_start = tl.load(CU_SEQLENS + pid_batch)
        seqlen_end = tl.load(CU_SEQLENS + pid_batch + 1)
        current_seqlen = seqlen_end - seqlen_start
        offset_b = seqlen_start * stride_x_seqlen
        offset_bo = seqlen_start * stride_out_seqlen

    X = X + offset_b + pid_head * stride_x_nheads
    OUT = OUT + offset_bo + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= current_seqlen:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < current_seqlen

    if IS_SEQLEN_OFFSETS_TENSOR:
        seqlen_offset = tl.load(SEQLEN_OFFSETS + pid_batch)
    else:
        seqlen_offset = SEQLEN_OFFSETS

    rk_half = tl.arange(0, BLOCK_K // 2)
    rk_full = tl.arange(0, BLOCK_K)

    if not INTERLEAVED:
        # Non-interleaved
        cos_offset = (rm[:, None] + seqlen_offset) * rotary_dim + rk_half[None, :]
        cos = tl.load(COS + cos_offset, 
                     mask=((rm[:, None] + seqlen_offset) < seqlen_ro) & (rk_half[None, :] < rotary_dim//2), 
                     other=1.0).to(tl.float32)
        sin = tl.load(SIN + cos_offset, 
                     mask=((rm[:, None] + seqlen_offset) < seqlen_ro) & (rk_half[None, :] < rotary_dim//2), 
                     other=0.0).to(tl.float32)
        
        if CONJUGATE:
            sin = -sin
        
        x0_offset = rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
        x0 = tl.load(X + x0_offset, mask=mask_m[:, None] & (rk_half[None, :] < rotary_dim//2), other=0.0).to(tl.float32)
        x1_offset = rm[:, None] * stride_x_seqlen + (rk_half[None, :] + rotary_dim//2) * stride_x_headdim
        x1 = tl.load(X + x1_offset, mask=mask_m[:, None] & (rk_half[None, :] < rotary_dim//2), other=0.0).to(tl.float32)
        
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        tl.store(OUT + x0_offset, y0, mask=mask_m[:, None] & (rk_half[None, :] < rotary_dim//2))
        tl.store(OUT + x1_offset, y1, mask=mask_m[:, None] & (rk_half[None, :] < rotary_dim//2))
        
        # Remaining dimensions
        if rotary_dim < BLOCK_K:
            rk_rem = tl.arange(rotary_dim, BLOCK_K)
            x_rem = tl.load(X + rm[:, None] * stride_x_seqlen + rk_rem[None, :] * stride_x_headdim,
                           mask=mask_m[:, None] & (rk_rem[None, :] < BLOCK_K), other=0.0)
            tl.store(OUT + rm[:, None] * stride_out_seqlen + rk_rem[None, :] * stride_out_headdim,
                    x_rem, mask=mask_m[:, None] & (rk_rem[None, :] < BLOCK_K))

    else:
        # Interleaved
        cos_offset = (rm[:, None] + seqlen_offset) * rotary_dim + (rk_full[None, :]//2)
        cos = tl.load(COS + cos_offset,
                     mask=((rm[:, None] + seqlen_offset) < seqlen_ro) & (rk_full[None, :] < rotary_dim),
                     other=1.0).to(tl.float32)
        sin = tl.load(SIN + cos_offset,
                     mask=((rm[:, None] + seqlen_offset) < seqlen_ro) & (rk_full[None, :] < rotary_dim),
                     other=0.0).to(tl.float32)

        x_offset = rm[:, None] * stride_x_seqlen + rk_full[None, :] * stride_x_headdim
        x = tl.load(X + x_offset, mask=mask_m[:, None] & (rk_full[None, :] < rotary_dim), other=0.0).to(tl.float32)

        if CONJUGATE:
            sin = -sin

        rk_even = (rk_full[None, :] % 2) == 0
        y = tl.where(rk_even, x * cos - x * sin.flip(1), x * sin + x * cos.flip(1))

        tl.store(OUT + x_offset, y, mask=mask_m[:, None] & (rk_full[None, :] < rotary_dim))


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
    seqlen_ro, rotary_dim = cos.shape
    
    assert rotary_dim <= headdim
    assert seqlen_ro >= seqlen
    assert cos.dtype == sin.dtype == x.dtype
    assert rotary_dim % 2 == 0
    
    if cu_seqlens is not None:
        assert cu_seqlens.dtype == torch.int32
    
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.dtype == torch.int32
        seqlen_offsets = seqlen_offsets.contiguous()
    
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = triton.next_power_of_2(rotary_dim)
    
    if cu_seqlens is None:
        grid_m = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads)
    else:
        grid_m = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), cu_seqlens.shape[0]-1, nheads)
    
    BLOCK_M = 4 if interleaved else 8
    
    rotary_kernel[grid_m](
        output,
        x,
        cos,
        sin,
        cu_seqlens,
        seqlen_offsets if isinstance(seqlen_offsets, torch.Tensor) else None,
        seqlen,
        rotary_dim,
        seqlen_ro,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_K=BLOCK_K,
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
        IS_VARLEN=cu_seqlens is not None,
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
