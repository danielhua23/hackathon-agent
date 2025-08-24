
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
    max_seqlen,
    stride_xb,
    stride_xh,
    stride_xm,
    stride_xk,
    stride_cosm,
    stride_cosk,
    stride_sinm,
    stride_sink,
    stride_outb,
    stride_outh,
    stride_outm,
    stride_outk,
    TOTAL_TOKENS,
    HEAD_NUM,
    HEAD_DIM: tl.constexpr,
    IS_INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    IS_INPLACE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    seq_start = 0
    seq_end = max_seqlen
    if CU_SEQLENS is not None:
        seq_start = tl.load(CU_SEQLENS + pid_batch)
        seq_end = tl.load(CU_SEQLENS + pid_batch + 1)
    else:
        seq_start = pid_batch * max_seqlen
        seq_end = (pid_batch + 1) * max_seqlen

    actual_seqlen = seq_end - seq_start
    if pid_m * BLOCK_M >= actual_seqlen:
        return

    if CU_SEQLENS is not None:
        batch_offset = 0
    else:
        batch_offset = pid_batch

    head_offset = pid_head
    d_half = HEAD_DIM // 2

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < actual_seqlen

    if IS_INTERLEAVED:
        for il in range(0, HEAD_DIM // 2):
            offs_k_cos_0 = il
            offs_k_cos_1 = il + d_half

            if CU_SEQLENS is not None:
                ptr_x_0 = X + seq_start * stride_xm + head_offset * stride_xh + offs_m[:, None] * stride_xm + offs_k_cos_0 * 2 * stride_xk + offs_n[None, :] * 2
                ptr_x_1 = X + seq_start * stride_xm + head_offset * stride_xh + offs_m[:, None] * stride_xm + offs_k_cos_0 * 2 * stride_xk + offs_n[None, :] * 2 + stride_xk
                ptr_cos = COS + offs_m[:, None] * stride_cosm + offs_k_cos_0 * stride_cosk
                ptr_sin = SIN + offs_m[:, None] * stride_sinm + offs_k_cos_0 * stride_sink
            else:
                ptr_x_0 = X + batch_offset * stride_xb + head_offset * stride_xh + offs_m[:, None] * stride_xm + offs_k_cos_0 * 2 * stride_xk + offs_n[None, :] * 2
                ptr_x_1 = X + batch_offset * stride_xb + head_offset * stride_xh + offs_m[:, None] * stride_xm + offs_k_cos_0 * 2 * stride_xk + offs_n[None, :] * 2 + stride_xk
                ptr_cos = COS + offs_m[:, None] * stride_cosm + offs_k_cos_0 * stride_cosk
                ptr_sin = SIN + offs_m[:, None] * stride_sinm + offs_k_cos_0 * stride_sink

            x0 = tl.load(ptr_x_0, mask=mask_m[:, None])
            x1 = tl.load(ptr_x_1, mask=mask_m[:, None])
            c = tl.load(ptr_cos, mask=mask_m[:, None])
            s = tl.load(ptr_sin, mask=mask_m[:, None])

            if CONJUGATE:
                tmp = x0 * c + x1 * s
                x1 = x1 * c - x0 * s
                x0 = tmp
            else:
                tmp = x0 * c - x1 * s
                x1 = x0 * s + x1 * c
                x0 = tmp

            if IS_INPLACE:
                tl.store(ptr_x_0, x0.to(ptr_x_0.type.element_ty), mask=mask_m[:, None])
                tl.store(ptr_x_1, x1.to(ptr_x_1.type.element_ty), mask=mask_m[:, None])
            else:
                if CU_SEQLENS is not None:
                    ptr_out_0 = OUT + seq_start * stride_outm + head_offset * stride_outh + offs_m[:, None] * stride_outm + offs_k_cos_0 * 2 * stride_outk + offs_n[None, :] * 2
                    ptr_out_1 = OUT + seq_start * stride_outm + head_offset * stride_outh + offs_m[:, None] * stride_outm + offs_k_cos_0 * 2 * stride_outk + offs_n[None, :] * 2 + stride_outk
                else:
                    ptr_out_0 = OUT + batch_offset * stride_outb + head_offset * stride_outh + offs_m[:, None] * stride_outm + offs_k_cos_0 * 2 * stride_outk + offs_n[None, :] * 2
                    ptr_out_1 = OUT + batch_offset * stride_outb + head_offset * stride_outh + offs_m[:, None] * stride_outm + offs_k_cos_0 * 2 * stride_outk + offs_n[None, :] 2 + stride_outk
                tl.store(ptr_out_0, x0.to(ptr_out_0.type.element_ty), mask=mask_m[:, None])
                tl.store(ptr_out_1, x1.to(ptr_out_1.type.element_ty), mask=mask_m[:, None])
    else:
        for ih in range(0, 2):
            if ih == 0:
                offs_k_start = 0
                offs_k_end = d_half
                offs_cos_k = 0
            else:
                offs_k_start = d_half
                offs_k_end = HEAD_DIM
                offs_cos_k = 1

            if CU_SEQLENS is not None:
                ptr_x_base = X + seq_start * stride_xm + head_offset * stride_xh
                ptr_cos_base = COS + offs_m[:, None] * stride_cosm + offs_cos_k * stride_cosk
                ptr_sin_base = SIN + offs_m[:, None] * stride_sinm + offs_cos_k * stride_sink
                ptr_out_base = OUT + seq_start * stride_outm + head_offset * stride_outh
            else:
                ptr_x_base = X + batch_offset * stride_xb + head_offset * stride_xh
                ptr_cos_base = COS + offs_m[:, None] * stride_cosm + offs_cos_k * stride_cosk
                ptr_sin_base = SIN + offs_m[:, None] * stride_sinm + offs_cos_k * stride_sink
                ptr_out_base = OUT + batch_offset * stride_outb + head_offset * stride_outh

            x0 = tl.load(ptr_x_base + offs_m[:, None] * stride_xm + (tl.arange(offs_k_start, offs_k_end)[None, :]) * stride_xk, mask=mask_m[:, None])
            x1 = tl.load(ptr_x_base + offs_m[:, None] * stride_xm + (tl.arange(offs_k_start + d_half, offs_k_end + d_half)[None, :]) * stride_xk, mask=mask_m[:, None])

            c = tl.load(ptr_cos_base)
            s = tl.load(ptr_sin_base)

            if CONJUGATE:
                tmp = x0 * c + x1 * s
                x1 = x1 * c - x0 * s
                x0 = tmp
            else:
                tmp = x0 * c - x1 * s
                x1 = x0 * s + x1 * c
                x0 = tmp

            if IS_INPLACE:
                tl.store(ptr_x_base + offs_m[:, None] * stride_xm + (tl.arange(offs_k_start, offs_k_end)[None, :]) * stride_xk, x0.to(ptr_x_base.type.element_ty), mask=mask_m[:, None])
                tl.store(ptr_x_base + offs_m[:, None] * stride_xm + (tl.arange(offs_k_start + d_half, offs_k_end + d_half)[None, :]) * stride_xk, x1.to(ptr_x_base.type.element_ty), mask=mask_m[:, None])
            else:
                tl.store(ptr_out_base + offs_m[:, None] * stride_outm + (tl.arange(offs_k_start, offs_k_end)[None, :]) * stride_outk, x0.to(ptr_out_base.type.element_ty), mask=mask_m[:, None])
                tl.store(ptr_out_base + offs_m[:, None] * stride_outm + (tl.arange(offs_k_start + d_half, offs_k_end + d_half)[None, :]) * stride_outk, x1.to(ptr_out_base.type.element_ty), mask=mask_m[:, None])


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int = 0,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
):
    assert x.dim() == 4
    batch, head_num, seqlen, headdim = x.shape

    if max_seqlen is None:
        if cu_seqlens is None:
            max_seqlen = seqlen
        else:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    BLOCK_M = max(16, min(128, triton.next_power_of_2(max_seqlen)))

    if inplace:
        out = x
    else:
        out = torch.empty_like(x)

    grid = (
        batch,
        head_num,
        triton.cdiv(max_seqlen, BLOCK_M),
    )

    rotary_kernel[grid](
        out,
        x,
        cos,
        sin,
        cu_seqlens,
        None,
        max_seqlen,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        cos.stride(0) if cos.dim() > 1 else 0,
        cos.stride(1) if cos.dim() > 1 else cos.stride(0),
        sin.stride(0) if sin.dim() > 1 else 0,
        sin.stride(1) if sin.dim() > 1 else sin.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        batch * seqlen,
        head_num,
        HEAD_DIM=headdim,
        IS_INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
        IS_INPLACE=inplace,
        BLOCK_M=BLOCK_M,
        BLOCK_N=1,
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
