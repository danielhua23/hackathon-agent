
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    X,
    COS,
    SIN,
    OUT,
    CU_SEQLENS,
    seqlen_offsets,
    stride_xb,
    stride_xh,
    stride_xm,
    stride_xk,
    stride_cosb,
    stride_cosh,
    stride_cosm,
    stride_cosk,
    stride_sinb,
    stride_sinh,
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
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if CU_SEQLENS is not None:
        cu_seqlens_start = tl.load(CU_SEQLENS + pid_batch)
        cu_seqlens_end = tl.load(CU_SEQLENS + pid_batch + 1)
        seqlen = cu_seqlens_end - cu_seqlens_start
    else:
        cu_seqlens_start = 0

    offset = tl.load(seqlen_offsets + pid_batch) if seqlen_offsets is not None else 0
    seqlen = seqlen - offset
    if pid_m >= seqlen:
        return

    rotary_dim_half = rotary_dim // 2
    cols = tl.arange(0, BLOCK_K)
    mask = cols < rotary_dim_half

    offset_m = pid_m + offset
    pos = offset_m.to(tl.int32)

    if interleaved:
        cos_offset = pos * stride_cosm + (cols * 2) * stride_cosk
        sin_offset = pos * stride_sinm + (cols * 2) * stride_sink
    else:
        cos_offset = pos * stride_cosm + cols * stride_cosk
        sin_offset = pos * stride_sinm + cols * stride_sink

    cos = tl.load(COS + cos_offset, mask=mask, other=0.0)
    sin = tl.load(SIN + sin_offset, mask=mask, other=0.0)

    x_offset = (
        pid_batch * stride_xb
        + pid_head * stride_xh
        + pid_m * stride_xm
    )

    if interleaved:
        x_col0 = x_offset + (cols * 2) * stride_xk
        x_col1 = x_offset + (cols * 2 + 1) * stride_xk
        x0 = tl.load(X + x_col0, mask=mask, other=0.0)
        x1 = tl.load(X + x_col1, mask=mask, other=0.0)
    else:
        x_col0 = x_offset + cols * stride_xk
        x_col1 = x_offset + (cols + rotary_dim_half) * stride_xk
        x0 = tl.load(X + x_col0, mask=mask, other=0.0)
        x1 = tl.load(X + x_col1, mask=mask, other=0.0)

    if conjugate:
        x1 = -x1

    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    out_offset = (
        pid_batch * stride_ob
        + pid_head * stride_oh
        + pid_m * stride_om
    )

    if interleaved:
        tl.store(OUT + out_offset + (cols * 2) * stride_ok, out0, mask=mask)
        tl.store(OUT + out_offset + (cols * 2 + 1) * stride_ok, out1, mask=mask)
    else:
        tl.store(OUT + out_offset + cols * stride_ok, out0, mask=mask)
        tl.store(OUT + out_offset + (cols + rotary_dim_half) * stride_ok, out1, mask=mask)

    # Copy non-rotary dimensions
    cols_rest_start = rotary_dim if not interleaved else rotary_dim * 2
    cols_rest_end = max_seqlen
    cols_rest = cols_rest_start + tl.arange(0, BLOCK_K)
    mask_rest = cols_rest < cols_rest_end

    if interleaved:
        x_rest_offset = x_offset + cols_rest * stride_xk
        out_rest_offset = out_offset + cols_rest * stride_ok
    else:
        x_rest_offset = x_offset + cols_rest * stride_xk
        out_rest_offset = out_offset + cols_rest * stride_ok

    x_rest = tl.load(X + x_rest_offset, mask=mask_rest, other=0.0)
    tl.store(OUT + out_rest_offset, x_rest, mask=mask_rest)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: torch.Tensor = None,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = None,
    interleaved: bool = False,
    in_place: bool = False,
    conjugate: bool = False,
):
    batch, head, seqlen, dim = x.shape
    rotary_dim = cos.shape[-1]
    assert cos.shape == sin.shape
    assert rotary_dim * 2 <= dim, "Rotary dim must be <= half of hidden size"

    if max_seqlen is None:
        if cu_seqlens is None:
            max_seqlen = seqlen
        else:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    BLOCK_M = 1
    BLOCK_K = max(rotary_dim, 32)

    grid = (batch, head, seqlen)

    if not in_place:
        out = torch.empty_like(x)
    else:
        out = x

    rotary_kernel[grid](
        x,
        cos,
        sin,
        out,
        cu_seqlens,
        seqlen_offsets,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        cos.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        sin.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        max_seqlen,
        rotary_dim,
        seqlen,
        interleaved,
        conjugate,
        BLOCK_M,
        BLOCK_K,
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
