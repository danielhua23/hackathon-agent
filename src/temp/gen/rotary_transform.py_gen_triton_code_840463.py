
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    X, COS, SIN, CU_SEQLENS, SEQLENS, OUT,
    stride_batch, stride_seqlen, stride_head, stride_dim,
    rotary_dim, max_seqlen, total_seqlens,
    nheads, seqlen_ro, interleaved, conj, BLOCK_SIZE_M: tl.constexpr,
    IS_EVEN_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if pid_batch >= stride_batch:
        return
    if pid_head >= nheads:
        return

    if CU_SEQLENS is not None:
        seq_start = tl.load(CU_SEQLENS + pid_batch)
        seq_end = tl.load(CU_SEQLENS + pid_batch + 1)
        seqlen_i = seq_end - seq_start
    else:
        seq_start = pid_batch * max_seqlen
        seqlen_i = tl.load(SEQLENS + pid_batch) if SEQLENS is not None else max_seqlen

    if pid_m >= seqlen_i:
        return

    offset_m = seq_start + pid_m

    k_id = tl.arange(0, rotary_dim // 2)
    rotary_dim_half = rotary_dim // 2

    for k in range(0, rotary_dim_half, BLOCK_SIZE_M):
        k_idx = k + tl.arange(0, BLOCK_SIZE_M)
        mask = k_idx < rotary_dim_half

        pos_m = pid_m

        cos_idx = pos_m * stride_seqlen + k_idx * 2
        cos_offset = COS + cos_idx
        cos0 = tl.load(cos_offset, mask=mask)
        cos1 = tl.load(cos_offset + 1, mask=mask)

        sin_idx = pos_m * stride_seqlen + k_idx * 2
        sin_offset = SIN + sin_idx
        sin0 = tl.load(sin_offset, mask=mask)
        sin1 = tl.load(sin_offset + 1, mask=mask)

        if interleaved:
            x_idx0 = offset_m * stride_seqlen + pid_head * stride_head + k_idx * 2
            x_idx1 = offset_m * stride_seqlen + pid_head * stride_head + k_idx * 2 + 1
            x0 = tl.load(X + x_idx0, mask=mask)
            x1 = tl.load(X + x_idx1, mask=mask)
            if conj:
                x1 = -x1
            out0 = x0 * cos0 - x1 * sin0
            out1 = x0 * sin1 + x1 * cos1
            tl.store(OUT + x_idx0, out0, mask=mask)
            tl.store(OUT + x_idx1, out1, mask=mask)
        else:
            x_idx0 = offset_m * stride_seqlen + pid_head * stride_head + k_idx
            x_idx1 = offset_m * stride_seqlen + pid_head * stride_head + k_idx + rotary_dim_half
            x0 = tl.load(X + x_idx0, mask=mask)
            x1 = tl.load(X + x_idx1, mask=mask)
            if conj:
                x1 = -x1
            out0 = x0 * cos0 - x1 * sin0
            out1 = x0 * sin1 + x1 * cos1
            tl.store(OUT + x_idx0, out0, mask=mask)
            tl.store(OUT + x_idx1, out1, mask=mask)

    if IS_EVEN_K:
        return

    k_offset = rotary_dim
    total_dim = stride_dim
    for d in range(rotary_dim, total_dim):
        idx = offset_m * stride_seqlen + pid_head * stride_head + d
        val = tl.load(X + idx)
        tl.store(OUT + idx, val)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_offset: int = 0,
    cu_seqlens: torch.Tensor = None,
    seqlens: torch.Tensor = None,
    rotary_dim: int = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
):
    batch, seqlen, nheads, headdim = x.shape
    rotary_dim = rotary_dim or headdim

    assert rotary_dim % 2 == 0
    assert headdim >= rotary_dim
    assert cos.shape == (seqlen, rotary_dim // 2 * 2)
    assert sin.shape == (seqlen, rotary_dim // 2 * 2)

    if not inplace:
        out = torch.empty_like(x)
    else:
        out = x

    BLOCK_SIZE_M = min(max(triton.next_power_of_2(rotary_dim // 2), 16), 64)

    grid = lambda META: (batch, nheads, triton.cdiv(seqlen, 1))

    rotary_kernel[grid](
        x, cos, sin, cu_seqlens, seqlens, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        rotary_dim, seqlen, batch * seqlen,
        nheads, seqlen, interleaved, conjugate, BLOCK_SIZE_M,
        IS_EVEN_K=(headdim == rotary_dim),
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
