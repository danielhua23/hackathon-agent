
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
    seqlens,
    stride_xb,
    stride_xh,
    stride_xm,
    stride_xk,
    stride_cos_m,
    stride_cos_k,
    stride_sin_m,
    stride_sin_k,
    stride_ob,
    stride_oh,
    stride_om,
    stride_ok,
    num_heads,
    rotary_dim,
    seqlen_offsets,
    interleaved: tl.constexpr,
    conjugate: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if CU_SEQLENS is not None:
        seqlen_start = tl.load(CU_SEQLENS + pid_batch)
        seqlen_end = tl.load(CU_SEQLENS + pid_batch + 1)
        seq_len = seqlen_end - seqlen_start
    else:
        seqlen_start = 0
        seq_len = tl.load(seqlens + pid_batch)

    if pid_m * BLOCK_M >= seq_len:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K // 2)

    if CU_SEQLENS is not None:
        pos = seqlen_start + offs_m
    else:
        pos = seqlen_offsets + offs_m

    mask_m = offs_m < seq_len
    mask_k = offs_k < rotary_dim // 2

    if not interleaved:
        x0_ptrs = (
            X
            + pid_batch * stride_xb
            + pid_head * stride_xh
            + offs_m[:, None] * stride_xm
            + offs_k[None, :] * 2 * stride_xk
        )
        x1_ptrs = x0_ptrs + stride_xk

        cos_ptrs = COS + pos[:, None] * stride_cos_m + offs_k[None, :] * stride_cos_k
        sin_ptrs = SIN + pos[:, None] * stride_sin_m + offs_k[None, :] * stride_sin_k

        x0 = tl.load(x0_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        x1 = tl.load(x1_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        cos = tl.load(cos_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        sin = tl.load(sin_ptrs, mask=mask_m[:, None] & mask_k[None, :])

        if conjugate:
            sin = -sin

        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

        out0_ptrs = (
            OUT
            + pid_batch * stride_ob
            + pid_head * stride_oh
            + offs_m[:, None] * stride_om
            + offs_k[None, :] * 2 * stride_ok
        )
        out1_ptrs = out0_ptrs + stride_ok

        tl.store(out0_ptrs, out0, mask=mask_m[:, None] & mask_k[None, :])
        tl.store(out1_ptrs, out1, mask=mask_m[:, None] & mask_k[None, :])
    else:
        x_real_ptrs = (
            X
            + pid_batch * stride_xb
            + pid_head * stride_xh
            + offs_m[:, None] * stride_xm
            + offs_k[None, :] * stride_xk * 2
        )
        x_imag_ptrs = x_real_ptrs + stride_xk

        cos_ptrs = COS + pos[:, None] * stride_cos_m + offs_k[None, :] * stride_cos_k
        sin_ptrs = SIN + pos[:, None] * stride_sin_m + offs_k[None, :] * stride_sin_k

        x_real = tl.load(x_real_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        x_imag = tl.load(x_imag_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        cos = tl.load(cos_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        sin = tl.load(sin_ptrs, mask=mask_m[:, None] & mask_k[None, :])

        if conjugate:
            x_imag = -x_imag

        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos

        out_real_ptrs = (
            OUT
            + pid_batch * stride_ob
            + pid_head * stride_oh
            + offs_m[:, None] * stride_om
            + offs_k[None, :] * stride_ok * 2
        )
        out_imag_ptrs = out_real_ptrs + stride_ok

        tl.store(out_real_ptrs, out_real, mask=mask_m[:, None] & mask_k[None, :])
        tl.store(out_imag_ptrs, out_imag, mask=mask_m[:, None] & mask_k[None, :])


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int = 0,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = None,
    interleaved: bool = False,
    conjugate: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    batch, seqlen, num_heads, head_dim = x.shape
    rotary_dim = cos.shape[-1]
    assert rotary_dim <= head_dim
    assert rotary_dim % 2 == 0
    assert cos.shape == (seqlen, rotary_dim)
    assert sin.shape == (seqlen, rotary_dim)

    BLOCK_K = 128
    BLOCK_M = 64

    grid = (batch, num_heads, triton.cdiv(seqlen, BLOCK_M))

    if cu_seqlens is not None:
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens.device == x.device
        max_seqlen = cu_seqlens.diff().max().item()

    if inplace:
        out = x
    else:
        out = torch.empty_like(x)

    rotary_kernel[grid](
        x,
        cos,
        sin,
        out,
        cu_seqlens,
        None if cu_seqlens is not None else torch.tensor([seqlen], dtype=torch.int32, device=x.device),
        x.stride(0),
        x.stride(2),
        x.stride(1),
        x.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        out.stride(3),
        num_heads,
        rotary_dim,
        seqlen_offsets,
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
