import torch
import triton
import triton.language as tl
import logging
from typing import Optional, Union

@triton.jit
def rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS, seqlen, nheads, rotary_dim, seqlen_ro, CACHE_KEY_SEQLEN, stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim, stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim, BLOCK_K: tl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr, IS_VARLEN: tl.constexpr, INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    rotary_dim_half = rotary_dim // 2
    if not IS_VARLEN:
        cur_seqlen = seqlen
        x_ptr = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        out_ptr = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        end_idx = tl.load(CU_SEQLENS + pid_batch + 1)
        cur_seqlen = end_idx - start_idx
        x_ptr = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        out_ptr = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < cur_seqlen
    if IS_SEQLEN_OFFSETS_TENSOR:
        offset = tl.load(SEQLEN_OFFSETS + pid_batch)
    else:
        offset = SEQLEN_OFFSETS
    rm_cs = rm + offset
    mask_m_cs = rm_cs < seqlen_ro
    if not INTERLEAVED:
        rk_half = tl.arange(0, BLOCK_K)
        mask_k_half = rk_half < rotary_dim_half
        cos_offs = rm_cs[:, None] * rotary_dim_half + rk_half[None, :]
        sin_offs = cos_offs
        cos = tl.load(COS + cos_offs, mask=mask_m_cs[:, None] & mask_k_half[None, :], other=1.0).to(tl.float32)
        sin = tl.load(SIN + sin_offs, mask=mask_m_cs[:, None] & mask_k_half[None, :], other=0.0).to(tl.float32)
        x0_offs = x_ptr + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
        x1_offs = x0_offs + rotary_dim_half * stride_x_headdim
        x0 = tl.load(x0_offs, mask=mask_m[:, None] & mask_k_half[None, :], other=0.0).to(tl.float32)
        x1 = tl.load(x1_offs, mask=mask_m[:, None] & mask_k_half[None, :], other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        out0_offs = out_ptr + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim
        out1_offs = out0_offs + rotary_dim_half * stride_out_headdim
        tl.store(out0_offs, y0, mask=mask_m[:, None] & mask_k_half[None, :])
        tl.store(out1_offs, y1, mask=mask_m[:, None] & mask_k_half[None, :])
    else:
        rk = tl.arange(0, BLOCK_K)
        mask_k = rk < rotary_dim
        rk_repeat = rk // 2
        cs_mask = rk_repeat[None, :] < rotary_dim_half
        cos_offs = rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :]
        sin_offs = cos_offs
        cos = tl.load(COS + cos_offs, mask=mask_m_cs[:, None] & cs_mask, other=1.0).to(tl.float32)
        sin = tl.load(SIN + sin_offs, mask=mask_m_cs[:, None] & cs_mask, other=0.0).to(tl.float32)
        rk_swap = rk + (rk + 1) % 2 * 2 - 1
        x0_offs = x_ptr + rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim
        x1_offs = x_ptr + rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim
        x0 = tl.load(x0_offs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        x1 = tl.load(x1_offs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        out_offs = out_ptr + rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim
        tl.store(out_offs, out, mask=mask_m[:, None] & mask_k[None, :])

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[torch.Tensor]=None, max_seqlen: Optional[int]=None, interleaved: bool=False, inplace: bool=False, conjugate: bool=False) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, 'If cu_seqlens is passed in, then max_seqlen must be passed'
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    assert rotary_dim <= headdim
    rotary_dim = rotary_dim * 2
    assert headdim <= 256, 'Only support headdim <= 256'
    assert seqlen_ro >= seqlen, 'seqlen_ro must be >= seqlen'
    assert cos.dtype == sin.dtype
    assert x.dtype == cos.dtype
    cos = cos.contiguous()
    sin = sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and (not inplace):
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    if interleaved:
        BLOCK_M = 4
    else:
        BLOCK_M = 8 if rotary_dim <= 64 else 4
    BLOCK_K = 32 if rotary_dim <= 32 else 64 if rotary_dim <= 64 else 128 if rotary_dim <= 128 else 256
    grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads)
    rotary_kernel[grid](output, x, cos, sin, cu_seqlens, seqlen_offsets, seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128, output.stride(0) if not is_varlen else 0, output.stride(-3), output.stride(-2), output.stride(-1), x.stride(0) if not is_varlen else 0, x.stride(-3), x.stride(-2), x.stride(-1), BLOCK_K, isinstance(seqlen_offsets, torch.Tensor), is_varlen, interleaved, conjugate, BLOCK_M)
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
