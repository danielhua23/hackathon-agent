import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.autotune(configs=[triton.Config({'BLOCK_M': 4, 'BLOCK_K': 32}, num_warps=2, num_stages=1), triton.Config({'BLOCK_M': 8, 'BLOCK_K': 32}, num_warps=2, num_stages=1), triton.Config({'BLOCK_M': 4, 'BLOCK_K': 64}, num_warps=4, num_stages=1), triton.Config({'BLOCK_M': 8, 'BLOCK_K': 64}, num_warps=4, num_stages=1), triton.Config({'BLOCK_M': 8, 'BLOCK_K': 128}, num_warps=4, num_stages=1), triton.Config({'BLOCK_M': 8, 'BLOCK_K': 256}, num_warps=8, num_stages=1)], key=['HEAD_DIM', 'ROTARY_DIM', 'INTERLEAVED'])
@triton.jit
def rotary_kernel(X, COS, SIN, OUT, CU_SEQLENS, SEQ_OFFSETS, stride_xb, stride_xh, stride_xm, stride_xd, stride_cos_m, stride_cos_d, stride_sin_m, stride_sin_d, stride_ob, stride_oh, stride_om, stride_od, nheads, rotary_dim, HEAD_DIM: tl.constexpr, seqlen, interleaved: tl.constexpr, conjugate: tl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr, IS_VARLEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    rotary_dim_half = rotary_dim // 2
    if IS_VARLEN:
        seq_start = tl.load(CU_SEQLENS + pid_batch).to(tl.int32)
        seq_end = tl.load(CU_SEQLENS + pid_batch + 1).to(tl.int32)
        cur_seqlen = seq_end - seq_start
    else:
        seq_start = 0
        cur_seqlen = seqlen
    if pid_m * BLOCK_M >= cur_seqlen:
        return
    BLOCK_K_ACT = min(BLOCK_K, rotary_dim_half)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, BLOCK_K_ACT)
    x_base = X + pid_batch * stride_xb + pid_head * stride_xh
    out_base = OUT + pid_batch * stride_ob + pid_head * stride_oh
    cos_base = COS
    sin_base = SIN
    if not IS_SEQLEN_OFFSETS_TENSOR:
        base_m_cs = rm + seq_start + seq_off
    else:
        seq_off_val = tl.load(SEQ_OFFSETS + pid_batch)
        base_m_cs = rm + seq_start + seq_off_val
    mask_m = rm < cur_seqlen
    if not interleaved:
        for k_offset in range(0, rotary_dim_half, BLOCK_K):
            k_cur = k_offset + rk_half
            mask_k = k_cur < rotary_dim_half
            cos_off = base_m_cs[:, None] * stride_cos_m + k_cur[None, :] * stride_cos_d
            cos = tl.load(cos_base + cos_off, mask=mask_m[:, None] & mask_k[None, :], other=1.0).to(tl.float32)
            sin_off = base_m_cs[:, None] * stride_sin_m + k_cur[None, :] * stride_sin_d
            sin = tl.load(sin_base + sin_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            if conjugate:
                sin = -sin
            x0_off = (rm[:, None] + seq_start) * stride_xm + k_cur[None, :] * stride_xd
            x0 = tl.load(x_base + x0_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            x1_off = (rm[:, None] + seq_start) * stride_xm + (k_cur + rotary_dim_half)[None, :] * stride_xd
            x1 = tl.load(x_base + x1_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            o0 = x0 * cos - x1 * sin
            o1 = x0 * sin + x1 * cos
            out0_off = (rm[:, None] + seq_start) * stride_om + k_cur[None, :] * stride_od
            tl.store(out_base + out0_off, o0, mask=mask_m[:, None] & mask_k[None, :])
            out1_off = (rm[:, None] + seq_start) * stride_om + (k_cur + rotary_dim_half)[None, :] * stride_od
            tl.store(out_base + out1_off, o1, mask=mask_m[:, None] & mask_k[None, :])
    else:
        for k_base in range(0, rotary_dim, 2 * BLOCK_K):
            k_even = 2 * k_base + 2 * rk_half
            k_odd = 2 * k_base + 2 * rk_half + 1
            mask_k = k_even < rotary_dim
            cos_off = base_m_cs[:, None] * stride_cos_m + (k_even // 2)[None, :] * stride_cos_d
            cos = tl.load(cos_base + cos_off, mask=mask_m[:, None] & mask_k[None, :], other=1.0).to(tl.float32)
            sin_off = base_m_cs[:, None] * stride_sin_m + (k_even // 2)[None, :] * stride_sin_d
            sin = tl.load(sin_base + sin_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            if conjugate:
                sin = -sin
            xe_off = (rm[:, None] + seq_start) * stride_xm + k_even[None, :] * stride_xd
            x0 = tl.load(x_base + xe_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            xo_off = (rm[:, None] + seq_start) * stride_xm + k_odd[None, :] * stride_xd
            x1 = tl.load(x_base + xo_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            out0 = x0 * cos - x1 * sin
            out1 = x0 * sin + x1 * cos
            oe_off = (rm[:, None] + seq_start) * stride_om + k_even[None, :] * stride_od
            tl.store(out_base + oe_off, out0, mask=mask_m[:, None] & mask_k[None, :])
            oo_off = (rm[:, None] + seq_start) * stride_om + k_odd[None, :] * stride_od
            tl.store(out_base + oo_off, out1, mask=mask_m[:, None] & mask_k[None, :])
    for d_offset in range(rotary_dim, HEAD_DIM, BLOCK_K):
        d_cur = d_offset + rk_half
        mask_d = d_cur < HEAD_DIM
        xt_off = (rm[:, None] + seq_start) * stride_xm + d_cur[None, :] * stride_xd
        x_tail = tl.load(x_base + xt_off, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        ot_off = (rm[:, None] + seq_start) * stride_om + d_cur[None, :] * stride_od
        tl.store(out_base + ot_off, x_tail, mask=mask_m[:, None] & mask_d[None, :])

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[torch.Tensor]=None, rotary_dim: Optional[int]=None, interleaved: bool=False, conjugate: bool=False, out: Optional[torch.Tensor]=None) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        total_seqlen, nheads, headdim = x.shape
        assert cu_seqlens.numel() > 1
        batch = cu_seqlens.numel() - 1
    seqlen_ro = cos.shape[0]
    rotary_dim_ = rotary_dim if rotary_dim is not None else cos.shape[1] * 2
    rotary_dim = min(rotary_dim_, headdim)
    assert rotary_dim % 2 == 0, 'rotary_dim must be even'
    assert rotary_dim <= headdim
    assert cos.shape == sin.shape
    assert x.dtype == cos.dtype == sin.dtype
    if isinstance(seqlen_offsets, int):
        seq_off_tensor = torch.tensor([seqlen_offsets], dtype=torch.int32, device=x.device).expand(batch)
    else:
        assert seqlen_offsets.shape == (batch,)
        seq_off_tensor = seqlen_offsets.contiguous()
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.shape == x.shape
        out.copy_(x)
    grid = lambda META: (triton.cdiv(x.shape[1] if not is_varlen else int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item()), META['BLOCK_M']), batch, nheads)
    rotary_kernel[grid](x, cos, sin, out, cu_seqlens, seq_off_tensor, x.stride(0), x.stride(2), x.stride(1), x.stride(3), cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1), out.stride(0), out.stride(2), out.stride(1), out.stride(3), nheads, rotary_dim, headdim, x.shape[1] if not is_varlen else 0, interleaved, conjugate, isinstance(seqlen_offsets, torch.Tensor), is_varlen)
    return out

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
