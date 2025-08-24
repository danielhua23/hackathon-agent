import torch
import triton
import triton.language as tl
from typing import Union, Optional

@triton.jit
def rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLENS, INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr, stride_xb, stride_xh, stride_xm, stride_xd, stride_cosb, stride_cosh, stride_cosm, stride_cosd, stride_sinb, stride_sinh, stride_sinm, stride_sind, stride_ob, stride_oh, stride_om, stride_od, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    if CU_SEQLENS is not None:
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - tl.load(CU_SEQLENS + pid_batch)
    else:
        seqlen = SEQLENS
    if pid_m * BLOCK_M >= seqlen:
        return
    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < HEAD_DIM // 2
    offs_base = pid_batch * stride_xb + pid_head * stride_xh
    offs_row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_row < seqlen
    cos_base = pid_batch * stride_cosb
    sin_base = pid_batch * stride_sinb
    if not INTERLEAVED:
        idx0 = offsets_d
        idx1 = offsets_d + HEAD_DIM // 2
        for m in range(BLOCK_M):
            mask_m_curr = mask_m[m]
            offs0 = offs_base + offs_row[m] * stride_xm + idx0 * stride_xd
            offs1 = offs_base + offs_row[m] * stride_xm + idx1 * stride_xd
            x0 = tl.load(X + offs0, mask=mask_d & mask_m_curr, other=0.0).to(tl.float32)
            x1 = tl.load(X + offs1, mask=mask_d & mask_m_curr, other=0.0).to(tl.float32)
            cidx = offs_row[m] * stride_cosm + idx0 * stride_cosd
            sidx = offs_row[m] * stride_sinm + idx0 * stride_sind
            cos = tl.load(COS + cidx, mask=mask_d & mask_m_curr, other=1.0).to(tl.float32)
            sin = tl.load(SIN + sidx, mask=mask_d & mask_m_curr, other=0.0).to(tl.float32)
            if CONJUGATE:
                sin = -sin
            y0 = x0 * cos - x1 * sin
            y1 = x0 * sin + x1 * cos
            tl.store(OUT + offs0, y0.to(X.dtype.element_ty), mask=mask_d & mask_m_curr)
            tl.store(OUT + offs1, y1.to(X.dtype.element_ty), mask=mask_d & mask_m_curr)
    else:
        idx_real = 2 * offsets_d
        idx_imag = 2 * offsets_d + 1
        mask_real = idx_real < HEAD_DIM
        mask_imag = idx_imag < HEAD_DIM
        for m in range(BLOCK_M):
            mask_m_curr = mask_m[m]
            offs_real = offs_base + offs_row[m] * stride_xm + idx_real * stride_xd
            offs_imag = offs_base + offs_row[m] * stride_xm + idx_imag * stride_xd
            real = tl.load(X + offs_real, mask=mask_real & mask_m_curr, other=0.0).to(tl.float32)
            imag = tl.load(X + offs_imag, mask=mask_imag & mask_m_curr, other=0.0).to(tl.float32)
            cidx = offs_row[m] * stride_cosm + offsets_d * stride_cosd
            sidx = offs_row[m] * stride_sinm + offsets_d * stride_sind
            cos = tl.load(COS + cidx, mask=mask_d & mask_m_curr, other=1.0).to(tl.float32)
            sin = tl.load(SIN + sidx, mask=mask_d & mask_m_curr, other=0.0).to(tl.float32)
            if CONJUGATE:
                imag = -imag
            out_real = real * cos - imag * sin
            out_imag = real * sin + imag * cos
            tl.store(OUT + offs_real, out_real.to(X.dtype.element_ty), mask=mask_real & mask_m_curr)
            tl.store(OUT + offs_imag, out_imag.to(X.dtype.element_ty), mask=mask_imag & mask_m_curr)

@triton.autotune(configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_D': 32}, num_stages=1, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_D': 64}, num_stages=1, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_D': 32}, num_stages=1, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_D': 64}, num_stages=1, num_warps=4)], key=['INTERLEAVED'])
@triton.jit
def tuned_rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLENS, INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr, stride_xb, stride_xh, stride_xm, stride_xd, stride_cosb, stride_cosh, stride_cosm, stride_cosd, stride_sinb, stride_sinh, stride_sinm, stride_sind, stride_ob, stride_oh, stride_om, stride_od, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLENS, INTERLEAVED, CONJUGATE, stride_xb, stride_xh, stride_xm, stride_xd, stride_cosb, stride_cosh, stride_cosm, stride_cosd, stride_sinb, stride_sinh, stride_sinm, stride_sind, stride_ob, stride_oh, stride_om, stride_od, BLOCK_M, BLOCK_D, HEAD_DIM)

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[torch.Tensor]=None, max_seqlen: Optional[int]=None, interleaved: bool=False, inplace: bool=False, conjugate: bool=False) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim
    assert x.dtype == cos.dtype == sin.dtype
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        seqlen_offsets = seqlen_offsets.int().contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and (not inplace):
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    cos = cos.contiguous()
    sin = sin.contiguous()
    grid = (batch, nheads, triton.cdiv(seqlen, 64))
    tuned_rotary_kernel[grid](output, x, cos, sin, None if cu_seqlens is None else cu_seqlens.int(), seqlen, stride_xb=x.stride(0) if not is_varlen else 1, stride_xh=x.stride(-2), stride_xm=x.stride(-3) if not is_varlen else 1, stride_xd=x.stride(-1), stride_cosb=cos.stride(0) if cos.ndim == 3 else 0, stride_cosh=cos.stride(1) if cos.ndim == 3 else 0, stride_cosm=cos.stride(-2), stride_cosd=cos.stride(-1), stride_sinb=sin.stride(0) if sin.ndim == 3 else 0, stride_sinh=sin.stride(1) if sin.ndim == 3 else 0, stride_sinm=sin.stride(-2), stride_sind=sin.stride(-1), stride_ob=output.stride(0) if not is_varlen else 1, stride_oh=output.stride(-2), stride_om=output.stride(-3) if not is_varlen else 1, stride_od=output.stride(-1), HEAD_DIM=headdim, INTERLEAVED=interleaved, CONJUGATE=conjugate)
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
