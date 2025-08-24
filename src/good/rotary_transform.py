import torch
import triton
import triton.language as tl
from typing import Union, Optional

@triton.jit
def rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS, seqlen, nheads, rotary_dim, seqlen_ro, CACHE_KEY_SEQLEN, stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim, stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim, BLOCK_K: tl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr, IS_VARLEN: tl.constexpr, INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
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
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)
    if not INTERLEAVED:
        x0_ptr = X + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
        x1_ptr = x0_ptr + rotary_dim_half * stride_x_headdim
        cos_ptr = COS + rm_cs[:, None] * rotary_dim_half + rk_half[None, :]
        sin_ptr = SIN + rm_cs[:, None] * rotary_dim_half + rk_half[None, :]
        cos = tl.load(cos_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x0 = tl.load(x0_ptr, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        out0_ptr = OUT + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim
        out1_ptr = out0_ptr + rotary_dim_half * stride_out_headdim
        tl.store(out0_ptr, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(out1_ptr, o1, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    else:
        rk_swap = rk + (rk + 1) % 2 * 2 - 1
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        x0_ptr = X + rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim
        x1_ptr = X + rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim
        cos_ptr = COS + rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :]
        sin_ptr = SIN + rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :]
        cos = tl.load(cos_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x0 = tl.load(x0_ptr, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        out = tl.where(rk[None, :] % 2 == 0, x0 * cos - x1 * sin, x0 * cos + x1 * sin)
        out_ptr = OUT + rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim
        tl.store(out_ptr, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[torch.Tensor]=None, max_seqlen: Optional[int]=None, interleaved: bool=False, inplace: bool=False, conjugate: bool=False) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        assert x.ndim == 4, 'Expected 4-D tensor [batch, seqlen, heads, dim] for non-varlen inputs'
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, 'If cu_seqlens is provided, max_seqlen must be specified'
        assert x.ndim == 3, 'Expected 3-D tensor [total_seqlen, heads, dim] for varlen inputs'
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim, 'rotary_dim must be <= headdim'
    assert cos.dtype == sin.dtype and x.dtype == cos.dtype
    assert seqlen_ro >= seqlen, 'seqlen_ro must be >= seqlen'
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in (torch.int32, torch.int64)
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert int(seqlen_offsets) + seqlen <= seqlen_ro
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and (not inplace):
        output[..., rotary_dim:].copy_(x[..., rotary_dim:]) if not is_varlen else output[:, rotary_dim:].copy_(x[:, rotary_dim:])
    BLOCK_K = 32 if rotary_dim <= 32 else 64 if rotary_dim <= 64 else 128 if rotary_dim <= 128 else 256
    BLOCK_M = 4 if interleaved else 8 if rotary_dim <= 64 else 4
    grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads)

    def stride_or_zero(tensor, idx, fixed=None):
        return tensor.stride(idx) if fixed is None else fixed
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](output, x, cos, sin, cu_seqlens, seqlen_offsets, seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128, stride_or_zero(output, -4, 0) if not is_varlen else 0, output.stride(-3), output.stride(-2), output.stride(-1), stride_or_zero(x, -4, 0) if not is_varlen else 0, x.stride(-3), x.stride(-2), x.stride(-1), BLOCK_K, isinstance(seqlen_offsets, torch.Tensor), is_varlen, interleaved, conjugate, BLOCK_M)
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
