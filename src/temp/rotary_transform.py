
import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLENS,
    SEQLEN_OFFSETS,
    max_seqlens,
    stride_outb,
    stride_outh,
    stride_outm,
    stride_outk,
    stride_xb,
    stride_xh,
    stride_xm,
    stride_xk,
    stride_cosb,
    stride_coss,
    stride_cosk,
    stride_sinb,
    stride_sins,
    stride_sink,
    rotary_dim,
    seqlen_offsets_ptr,
    conjugate: tl.constexpr,
    interleaved: tl.constexpr,
    seqlen_ro: tl.constexpr,
    stride_outg: tl.constexpr,
    stride_xg: tl.constexpr,
    max_sequence_length: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if CU_SEQLENS is not None:
        cu_seqlen_batch = pid_batch
        cu_seqlen_prev = tl.load(CU_SEQLENS + cu_seqlen_batch)
        cu_seqlen_curr = tl.load(CU_SEQLENS + cu_seqlen_batch + 1)
        seqlen = cu_seqlen_curr - cu_seqlen_prev
        offset_m_start = cu_seqlen_prev + pid_m * BLOCK_M
    else:
        seqlen_curr = tl.load(SEQLENS + pid_batch)
        seqlen = seqlen_curr
        offset_m_start = pid_m * BLOCK_M

    if seqlen <= 0:
        return

    offset_k = tl.arange(0, BLOCK_K)
    m_offset = offset_m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offset < seqlen

    if rotary_dim != -1:
        k_mask = offset_k < rotary_dim
    else:
        k_mask = offset_k < stride_outk

    start_m = m_offset[:, None]
    start_k = offset_k[None, :]

    if SEQLEN_OFFSETS is not None and seqlen_offsets_ptr:
        seqlen_offset = tl.load(SEQLEN_OFFSETS + pid_batch)
    else:
        seqlen_offset = 0

    pos_m = start_m + seqlen_offset
    pos_cos = pos_m % max_seqlens
    pos_sin = pos_m % max_seqlens

    cos_ptr = COS + pos_cos[:, None] * stride_cosb + start_k * stride_cosk
    sin_ptr = SIN + pos_sin[:, None] * stride_sinb + start_k * stride_sink

    cos = tl.load(cos_ptr, mask=m_mask[:, None] & k_mask[None, :])
    sin = tl.load(sin_ptr, mask=m_mask[:, None] & k_mask[None, :])

    x_ptr0 = X + pid_batch * stride_xb + pid_head * stride_xh + start_m * stride_xm + start_k * stride_xk
    x_ptr1 = X + pid_batch * stride_xb + pid_head * stride_xh + start_m * stride_xm + (start_k + 1) * stride_xk

    x0 = tl.load(x_ptr0, mask=m_mask[:, None] & k_mask[None, :])
    x1 = tl.load(x_ptr1, mask=m_mask[:, None] & k_mask[None, :])

    if interleaved:
        o_real = x0 * cos - x1 * sin
        o_imag = x1 * cos + x0 * sin
        if conjugate:
            o_imag = -o_imag
        out_ptr0 = OUT + pid_batch * stride_outb + pid_head * stride_outh + start_m * stride_outm + start_k * stride_outk
        out_ptr1 = OUT + pid_batch * stride_outb + pid_head * stride_outh + start_m * stride_outm + (start_k + 1) * stride_outk
        tl.store(out_ptr0, o_real, mask=m_mask[:, None] & k_mask[None, :])
        tl.store(out_ptr1, o_imag, mask=m_mask[:, None] & k_mask[None, :])
    else:
        cos_mask = start_k % 2 == 0
        sin_mask = start_k % 2 == 1
        x_even = tl.where(cos_mask, x0, 0.0)
        x_odd = tl.where(sin_mask, x0, 0.0)
        o_real = x_even * cos[None, :] - x_odd * sin[None, :]
        if conjugate:
            o_imag = x_odd * cos[None, :] + x_even * sin[None, :]
        else:
            o_imag = x_odd * cos[None, :] + x_even * sin[None, :]
        out_ptr0 = OUT + pid_batch * stride_outb + pid_head * stride_outh + start_m * stride_outm + start_k * stride_outk
        tl.store(out_ptr0, tl.where(cos_mask[None, :], o_real, o_imag), mask=m_mask[:, None] & k_mask[None, :])


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int = 0,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = None,
    interleaved: bool = False,
    in_place: bool = False,
    conjugate: bool = False,
    seqlen_ro: int = None,
) -> torch.Tensor:
    assert x.dim() >= 3
    batch = x.shape[0]
    seqlen = x.shape[-2]
    head = x.shape[-3] if x.dim() >= 4 else 1
    dim = x.shape[-1]
    rotary_dim = cos.shape[-1] if cos is not None else dim

    if max_seqlen is None:
        max_seqlen = seqlen
    assert cos is not None and sin is not None
    assert cos.dim() == 3 and sin.dim() == 3
    cos = cos.view(-1, max_seqlen, rotary_dim)
    sin = sin.view(-1, max_seqlen, rotary_dim)

    stride_outb = x.stride(0) if x.dim() >= 3 else 0
    stride_outh = x.stride(-3) if x.dim() >= 4 else 0
    stride_outm = x.stride(-2)
    stride_outk = x.stride(-1)
    stride_xb = x.stride(0) if x.dim() >= 3 else 0
    stride_xh = x.stride(-3) if x.dim() >= 4 else 0
    stride_xm = x.stride(-2)
    stride_xk = x.stride(-1)
    stride_cosb = cos.stride(0)
    stride_coss = cos.stride(1)
    stride_cosk = cos.stride(2)
    stride_sinb = sin.stride(0)
    stride_sins = sin.stride(1)
    stride_sink = sin.stride(2)

    seqlen_offsets_tensor = torch.tensor([seqlen_offsets], dtype=torch.int32, device=x.device) if isinstance(seqlen_offsets, int) else seqlen_offsets

    if in_place:
        out = x
    else:
        out = torch.empty_like(x)

    grid = (batch, head, (seqlen + 63) // 64)

    rotary_kernel[grid](
        out,
        x,
        cos,
        sin,
        cu_seqlens,
        torch.tensor([seqlen], dtype=torch.int32, device=x.device) if cu_seqlens is None else None,
        seqlen_offsets_tensor,
        torch.tensor([max_seqlen], dtype=torch.int32, device=x.device) if max_seqlen is not None else torch.tensor([seqlen], dtype=torch.int32, device=x.device),
        stride_outb,
        stride_outh,
        stride_outm,
        stride_outk,
        stride_xb,
        stride_xh,
        stride_xm,
        stride_xk,
        stride_cosb,
        stride_coss,
        stride_cosk,
        stride_sinb,
        stride_sins,
        stride_sink,
        rotary_dim,
        seqlen_offsets_tensor is not None,
        conjugate=conjugate,
        interleaved=interleaved,
        seqlen_ro=seqlen_ro if seqlen_ro is not None else seqlen,
        stride_outg=1,
        stride_xg=1,
        max_sequence_length=max_seqlen if max_seqlen is not None else seqlen,
        BLOCK_M=64,
        BLOCK_K=rotary_dim,
    )

    if out.dim() == 2:
        out = out.unsqueeze(0)
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
