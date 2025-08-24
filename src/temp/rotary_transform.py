
import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def rotary_kernel(
    OUT, X, COS, SIN,
    CU_SEQLENS,
    stride_x_batch, stride_x_head, stride_x_m, stride_x_k,
    stride_out_batch, stride_out_head, stride_out_m, stride_out_k,
    stride_cos_m, stride_cos_k,
    stride_sin_m, stride_sin_k,
    SEQLEN_OFFSETS,
    TOTAL_HEADS,
    INITIAL_HEAD_INDEX,
    BLOCK_K: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_batch = tl.program_id(2)

    initial_head = tl.load(INITIAL_HEAD_INDEX)
    total_heads = tl.load(TOTAL_HEADS)
    cur_head = initial_head + pid_head
    if cur_head >= total_heads:
        return

    head_dim = BLOCK_K * 2
    curr_dtype = X.type.element_ty

    offs_k = tl.arange(0, BLOCK_K)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    if IS_VARLEN:
        seq_beg = tl.load(CU_SEQLENS + pid_batch)
        seq_end = tl.load(CU_SEQLENS + pid_batch + 1)
        seqlen_i = seq_end - seq_beg
        mask_m = offs_m < seqlen_i
    else:
        seq_beg = 0
        seqlen_i = stride_x_m
        mask_m = offs_m < seqlen_i

    base_x = pid_batch * stride_x_batch + cur_head * stride_x_head
    base_out = pid_batch * stride_out_batch + cur_head * stride_out_head
    base_cos = offs_m
    seqlen_offset = tl.load(SEQLEN_OFFSETS + pid_batch) if SEQLEN_OFFSETS else 0
    base_cos = base_cos + seqlen_offset

    mask_k = offs_k < (head_dim // 2)
    cos = tl.load(
        COS + base_cos[:, None] * stride_cos_m + offs_k[None, :] * stride_cos_k,
        mask=mask_m[:, None] & mask_k[None, :], other=0.0
    ).to(tl.float32)
    sin = tl.load(
        SIN + base_cos[:, None] * stride_sin_m + offs_k[None, :] * stride_sin_k,
        mask=mask_m[:, None] & mask_k[None, :], other=0.0
    ).to(tl.float32)

    for m_step in range(BLOCK_M):
        if not mask_m[m_step]:
            continue
        curr_m = pid_m * BLOCK_M + m_step
        c = cos[m_step, :]
        s = sin[m_step, :]

        if INTERLEAVED:
            offs_x0 = base_x + curr_m * stride_x_m + offs_k * 2 * stride_x_k
            offs_x1 = offs_x0 + stride_x_k
            x0 = tl.load(X + offs_x0, mask=mask_k, other=0.0).to(tl.float32)
            x1 = tl.load(X + offs_x1, mask=mask_k, other=0.0).to(tl.float32)
        else:
            offs_x0 = base_x + curr_m * stride_x_m + offs_k * stride_x_k
            offs_x1 = base_x + curr_m * stride_x_m + (BLOCK_K + offs_k) * stride_x_k
            x0 = tl.load(X + offs_x0, mask=mask_k, other=0.0).to(tl.float32)
            x1 = tl.load(X + offs_x1, mask=mask_k, other=0.0).to(tl.float32)

        if CONJUGATE:
            x1 = -x1

        out0 = x0 * c - x1 * s
        out1 = x0 * s + x1 * c

        tl.store(OUT + offs_x0, out0.to(curr_dtype), mask=mask_k)
        tl.store(OUT + offs_x1, out1.to(curr_dtype), mask=mask_k)


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
    assert x.dtype in {torch.float16, torch.bfloat16, torch.float32}

    if cu_seqlens is None:
        batch, seqlen, nheads, headdim = x.shape
        is_varlen = False
        max_seqlen = seqlen
    else:
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.size(0) - 1
        seqlen = max_seqlen
        is_varlen = True

    seqlen_ro, rotary_dim_over2 = cos.shape
    assert sin.shape == cos.shape and headdim % 2 == 0
    rotary_dim = rotary_dim_over2 * 2
    assert rotary_dim <= headdim

    BLOCK_K = headdim // 2
    BLOCK_M = 64

    if isinstance(seqlen_offsets, int):
        seqlen_offsets_tensor = torch.tensor([seqlen_offsets], dtype=torch.int32, device=x.device)
    else:
        seqlen_offsets_tensor = seqlen_offsets.to(torch.int32)

    if inplace:
        out = x
    else:
        out = torch.empty_like(x)

    if rotary_dim < headdim and not inplace:
        out[..., rotary_dim:].copy_(x[..., rotary_dim:])

    total_heads = torch.tensor([nheads], dtype=torch.int32, device=x.device)
    initial_head = torch.tensor([0], dtype=torch.int32, device=x.device)

    grid = lambda META: (triton.cdiv(max_seqlen, META["BLOCK_M"]), nheads, batch)  

    rotary_kernel[grid](
        out, x, cos, sin,
        cu_seqlens,
        x.stride(0), x.stride(2), x.stride(1), x.stride(3),
        out.stride(0), out.stride(2), out.stride(1), out.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        seqlen_offsets_tensor,
        total_heads, initial_head,
        BLOCK_K=BLOCK_K,
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
        IS_VARLEN=is_varlen,
        BLOCK_M=BLOCK_M,
        EVEN_K=True,
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
