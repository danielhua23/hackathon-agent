
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
    SEQLENS,
    stride_xbatch,
    stride_xhead,
    stride_xm,
    stride_xk,
    stride_cos_m,
    stride_cos_k,
    stride_sin_m,
    stride_sin_k,
    stride_obatch,
    stride_ohead,
    stride_om,
    stride_ok,
    TOTAL_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_SEQLEN: tl.constexpr,
    IS_VARIABLE_L: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, BLOCK_K)

    if IS_VARIABLE_L:
        b_start = 0 if pid_batch == 0 else tl.load(CU_SEQLENS + pid_batch - 1)
        b_end = tl.load(CU_SEQLENS + pid_batch)
        seqlen = b_end - b_start
    else:
        b_start = pid_batch * MAX_SEQLEN
        seqlen = MAX_SEQLEN

    mask_m = offsets_m < seqlen
    mask_k_half = offsets_k < (HEAD_DIM // 2)

    full_offsets_m = b_start + offsets_m
    full_mask_m = full_offsets_m < TOTAL_TOKENS

    cos_ptrs = COS + full_offsets_m * stride_cos_m + offsets_k * stride_cos_k
    sin_ptrs = SIN + full_offsets_m * stride_sin_m + offsets_k * stride_sin_k

    if INTERLEAVED:
        x_offsets_k = offsets_k * 2
        x_offsets_k2 = offsets_k * 2 + 1
    else:
        x_offsets_k = offsets_k
        x_offsets_k2 = offsets_k + (HEAD_DIM // 2)

    x_ptrs = X + full_offsets_m * stride_xm + pid_head * stride_xhead + x_offsets_k * stride_xk
    x2_ptrs = X + full_offsets_m * stride_xm + pid_head * stride_xhead + x_offsets_k2 * stride_xk

    x1 = tl.load(x_ptrs, mask=full_mask_m[:, None] & mask_k_half[None, :])
    x2 = tl.load(x2_ptrs, mask=full_mask_m[:, None] & mask_k_half[None, :])

    cos = tl.load(cos_ptrs, mask=full_mask_m[:, None] & mask_k_half[None, :])
    sin = tl.load(sin_ptrs, mask=full_mask_m[:, None] & mask_k_half[None, :])

    if CONJUGATE:
        x2_rot = -x2
    else:
        x2_rot = x2

    out1 = x1 * cos - x2_rot * sin
    out2 = x1 * sin + x2 * cos

    out_ptrs = OUT + full_offsets_m * stride_om + pid_head * stride_ohead
    tl.store(out_ptrs + x_offsets_k * stride_ok, out1, mask=full_mask_m[:, None] & mask_k_half[None, :])
    tl.store(out_ptrs + x_offsets_k2 * stride_ok, out2, mask=full_mask_m[:, None] & mask_k_half[None, :])


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int = 0,
    cu_seqlens: torch.Tensor = None,
    *,
    inplace: bool = False,
    interleaved: bool = False,
    conj: bool = False,
):
    batch, seqlen, nheads, headdim = x.shape
    assert headdim <= 1024
    assert cos.shape == sin.shape == (seqlen, headdim // 2)
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]

    if inplace:
        out = x
    else:
        out = torch.empty_like(x)

    BLOCK_M = 1
    while BLOCK_M * nheads * 4 * x.element_size() < 32768 and BLOCK_M * 2 <= seqlen:
        BLOCK_M *= 2
    BLOCK_K = min(triton.next_power_of_2(headdim // 2), 64)

    grid = lambda META: (batch, nheads, triton.cdiv(seqlen, META["BLOCK_M"]))

    TOTAL_TOKENS = batch * seqlen
    max_seqlen = seqlen
    is_variable_l = cu_seqlens is not None
    HEAD_DIM = headdim

    rotary_kernel[grid](
        x,
        cos,
        sin,
        out,
        cu_seqlens,
        seqlen,
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
        TOTAL_TOKENS=TOTAL_TOKENS,
        HEAD_DIM=HEAD_DIM,
        MAX_SEQLEN=max_seqlen,
        IS_VARIABLE_L=is_variable_l,
        INTERLEAVED=interleaved,
        CONJUGATE=conj,
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
