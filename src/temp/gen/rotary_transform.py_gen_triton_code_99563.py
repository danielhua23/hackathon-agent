
import torch
import triton
import triton.language as tl
from typing import Union, Optional


@triton.jit
def rotary_kernel(
    X,
    COS,
    SIN,
    OUT,
    CU_SEQLENS,
    SEQLEN_OFFSETS,
    stride_xb,
    stride_xh,
    stride_xm,
    stride_xk,
    stride_cosm,
    stride_cosk,
    stride_sinm,
    stride_sink,
    stride_ob,
    stride_oh,
    stride_om,
    stride_ok,
    max_seqlen,
    rotary_dim,
    seqlen,
    interleaved: tl.constexpr,
    conjugate: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    if CU_SEQLENS is not None:
        seq_start = tl.load(CU_SEQLENS + pid_b)
        seq_end   = tl.load(CU_SEQLENS + pid_b + 1)
        current_seqlen = seq_end - seq_start
    else:
        current_seqlen = seqlen
        seq_start = 0

    if SEQLEN_OFFSETS is not None:
        offset = tl.load(SEQLEN_OFFSETS + pid_b).to(tl.int32)
    else:
        offset = 0

    if pid_m >= current_seqlen:
        return

    rotary_dim_half = rotary_dim // 2
    cols = tl.arange(0, BLOCK_K)
    mask = cols < rotary_dim_half

    pos = seq_start + pid_m + offset

    # load cos/sin
    cos_ptr = COS + pos * stride_cosm
    sin_ptr = SIN + pos * stride_sinm
    cos_val = tl.load(cos_ptr + cols * stride_cosk, mask=mask, other=1.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + cols * stride_sink, mask=mask, other=0.0).to(tl.float32)

    x_base = pid_b * stride_xb + pid_h * stride_xh + pid_m * stride_xm
    out_base = pid_b * stride_ob + pid_h * stride_oh + pid_m * stride_om

    if interleaved:
        even_ptrs = x_base + (cols * 2) * stride_xk
        odd_ptrs  = x_base + (cols * 2 + 1) * stride_xk
        x0 = tl.load(even_ptrs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(odd_ptrs,  mask=mask, other=0.0).to(tl.float32)

        if conjugate:
            x1 = -x1

        o0 = x0 * cos_val - x1 * sin_val
        o1 = x0 * sin_val + x1 * cos_val

        tl.store(out_base + (cols * 2) * stride_ok,     o0, mask=mask)
        tl.store(out_base + (cols * 2 + 1) * stride_ok, o1, mask=mask)
    else:
        left_ptrs  = x_base + cols * stride_xk
        right_ptrs = x_base + (cols + rotary_dim_half) * stride_xk
        x0 = tl.load(left_ptrs,  mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(right_ptrs, mask=mask, other=0.0).to(tl.float32)

        if conjugate:
            x1 = -x1

        o0 = x0 * cos_val - x1 * sin_val
        o1 = x0 * sin_val + x1 * cos_val

        tl.store(out_base + cols * stride_ok,               o0, mask=mask)
        tl.store(out_base + (cols + rotary_dim_half) * stride_ok, o1, mask=mask)

    cols_rest = rotary_dim + tl.arange(0, BLOCK_K)
    mask_rest = cols_rest < max_seqlen
    if mask_rest.any():
        src_rest = x_base + cols_rest * stride_xk
        dst_rest = out_base + cols_rest * stride_ok
        val_rest = tl.load(src_rest, mask=mask_rest, other=0.0)
        tl.store(dst_rest, val_rest, mask=mask_rest)


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
    if cu_seqlens is None:
        batch, nheads, seqlen, headdim = x.shape
    else:
        assert x.ndim == 3
        tot_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        if max_seqlen is None:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        seqlen = max_seqlen

    seqlen_ro, halfdim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim = halfdim * 2
    assert rotary_dim <= headdim
    assert x.dtype == cos.dtype == sin.dtype

    cos = cos.contiguous()
    sin = sin.contiguous()

    if isinstance(seqlen_offsets, int):
        seqlen_offsets_tensor = None
    else:
        assert seqlen_offsets.shape == (batch,)
        seqlen_offsets = seqlen_offsets.contiguous()
        seqlen_offsets_tensor = seqlen_offsets

    out = torch.empty_like(x) if not inplace else x

    # only copy non-rotary tails if not inplace
    if rotary_dim < headdim and not inplace:
        if cu_seqlens is None:
            out[..., rotary_dim:] = x[..., rotary_dim:]
        else:
            out[:, :, rotary_dim:] = x[:, :, rotary_dim:]

    BLOCK_M = 1
    BLOCK_K = triton.next_power_of_2(rotary_dim) // 2
    if cu_seqlens is None:
        grid = (batch, nheads, seqlen)
    else:
        grid = (batch, nheads, max_seqlen)

    rotary_kernel[grid](
        x, cos, sin, out,
        cu_seqlens,
        seqlen_offsets_tensor,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        headdim,
        rotary_dim,
        seqlen,
        interleaved,
        conjugate,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
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
