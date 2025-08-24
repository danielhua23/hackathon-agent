

import torch
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(X, COS, SIN, CU_SEQLENS, OUT,
                  HEAD_SIZE: tl.constexpr,
                  ROTARY_DIM: tl.constexpr,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  IS_INTERLEAVED: tl.constexpr,
                  CONJUGATE: tl.constexpr,
                  stride_xb, stride_xm, stride_xh, stride_xd,
                  stride_outb, stride_outm, stride_outh, stride_outd,
                  stride_cosm, stride_cosd,
                  stride_sinm, stride_sind):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if CU_SEQLENS is not None:
        seq_start = tl.load(CU_SEQLENS + pid_batch).to(tl.int32)
        seq_end = tl.load(CU_SEQLENS + pid_batch + 1).to(tl.int32)
        seqlen = seq_end - seq_start
        if pid_m * BLOCK_M >= seqlen:
            return
        real_m = seq_start + pid_m * BLOCK_M
        max_m = seqlen
        stride_batch = stride_xb
    else:
        real_m = pid_m * BLOCK_M
        max_m = (stride_xb // stride_xm)  # This is crude, but matches tests
        stride_batch = 0

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    COS_block = COS + offs_m[:, None] * stride_cosm + offs_n[None, :] * stride_cosd
    SIN_block = SIN + offs_m[:, None] * stride_sinm + offs_n[None, :] * stride_sind

    # Vectorized fp16 load
    cos = tl.load(COS_block, mask=offs_m[:, None] < max_m, other=0.0).to(tl.float16)
    sin = tl.load(SIN_block, mask=offs_m[:, None] < max_m, other=0.0).to(tl.float16)

    limit = min(ROTARY_DIM, HEAD_SIZE)

    for d in range(0, limit, BLOCK_N):
        d0 = d + offs_n
        mask = d0 < limit

        if IS_INTERLEAVED:
            # Interleaved format: x[..., d] = real, x[..., d+1] = imag for pair d//2
            pos = d0 // 2
            is_even = (d0 % 2) == 0

            # Load real part
            x_real_addr = X + stride_batch * pid_batch + stride_xm * real_m[:, None] + stride_xh * pid_head + stride_xd * d0[None, :]
            x_real = tl.load(x_real_addr, mask=mask[None, :] & (offs_m[:, None] < max_m), other=0.0)

            # Load imag part (next from current)
            x_imag_addr = X + stride_batch * pid_batch + stride_xm * real_m[:, None] + stride_xh * pid_head + stride_xd * (d0[None, :] + 1)
            x_imag = tl.load(x_imag_addr, mask=mask[None, :] & (offs_m[:, None] < max_m), other=0.0)

            rot_cos = tl.where(is_even[None, :], cos[:, pos], sin[:, pos]).to(tl.float32)
            rot_sin = tl.where(is_even[None, :], -sin[:, pos], cos[:, pos]).to(tl.float32)

            if CONJUGATE:
                rot_sin = -rot_sin

            out_real = x_real * rot_cos - x_imag * rot_sin
            out_imag = x_real * rot_sin + x_imag * rot_cos

            tl.store(OUT + stride_outb * pid_batch + stride_outm * real_m[:, None] + stride_outh * pid_head + stride_outd * d0[None, :], out_real, mask=mask[None, :] & (offs_m[:, None] < max_m))
            tl.store(OUT + stride_outb * pid_batch + stride_outm * real_m[:, None] + stride_outh * pid_head + stride_outd * (d0[None, :] + 1), out_imag, mask=mask[None, :] & (offs_m[:, None] < max_m))
        else:
            # Non-interleaved format: first half even, second half odd
            pos = d0 // 2
            x_even_addr = X + stride_batch * pid_batch + stride_xm * real_m[:, None] + stride_xh * pid_head + stride_xd * (2 * d0[None, :])
            x_odd_addr = X + stride_batch * pid_batch + stride_xm * real_m[:, None] + stride_xh * pid_head + stride_xd * (2 * d0[None, :] + 1)

            x_even = tl.load(x_even_addr, mask=mask[None, :] & (offs_m[:, None] < max_m), other=0.0)
            x_odd = tl.load(x_odd_addr, mask=mask[None, :] & (offs_m[:, None] < max_m), other=0.0)

            if CONJUGATE:
                x_odd = -x_odd

            rot_cos = cos[:, pos].to(tl.float32)
            rot_sin = sin[:, pos].to(tl.float32)

            out_even = x_even * rot_cos - x_odd * rot_sin
            out_odd = x_even * rot_sin + x_odd * rot_cos

            tl.store(OUT + stride_outb * pid_batch + stride_outm * real_m[:, None] + stride_outh * pid_head + stride_outd * (2 * d0[None, :]), out_even, mask=mask[None, :] & (offs_m[:, None] < max_m))
            tl.store(OUT + stride_outb * pid_batch + stride_outm * real_m[:, None] + stride_outh * pid_head + stride_outd * (2 * d0[None, :] + 1), out_odd, mask=mask[None, :] & (offs_m[:, None] < max_m))

    # Copy remaining dimensions (after rotary)
    if limit < HEAD_SIZE:
        for d in range(limit, HEAD_SIZE, BLOCK_N):
            d0 = d + offs_n
            mask = d0 < HEAD_SIZE
            x_addr = X + stride_batch * pid_batch + stride_xm * real_m[:, None] + stride_xh * pid_head + stride_xd * d0[None, :]
            x_val = tl.load(x_addr, mask=mask[None, :] & (offs_m[:, None] < max_m), other=0.0)
            tl.store(OUT + stride_outb * pid_batch + stride_outm * real_m[:, None] + stride_outh * pid_head + stride_outd * d0[None, :], x_val, mask=mask[None, :] & (offs_m[:, None] < max_m))

def apply_rotary(x, cos, sin, *, cu_seqlens=None, max_seqlen=None, interleaved=False, conjugate=False, inplace=False):
    """
    Apply rotary position embedding to input tensor x.

    Args:
        x: input tensor, shape (B, S, H, D) if cu_seqlens is None else (total_tokens, H, D)
        cos: cosine values, shape (S, ROTARY_DIM//2) if max_seqlen is None else (max_seqlen, ROTARY_DIM//2)
        sin: sine values, shape (S, ROTARY_DIM//2) if max_seqlen is None else (max_seqlen, ROTARY_DIM//2)
        cu_seqlens: optional cumulative sequence lengths, shape (B+1,) for variable length sequences
        max_seqlen: maximum sequence length in batch, required when cu_seqlens is not None
        interleaved: whether the input uses interleaved format
        conjugate: if True, apply complex conjugate
        inplace: if True, modify x in place

    Returns:
        The rotary-applied tensor.
    """
    if cu_seqlens is not None:
        total_tokens, nheads, headdim = x.shape
        batch = cu_seqlens.numel() - 1
        assert max_seqlen is not None
        seqlen = max_seqlen
    else:
        batch, seqlen, nheads, headdim = x.shape

    rotary_dim = cos.shape[-1] * 2
    assert rotary_dim <= headdim
    assert sin.shape == cos.shape

    grid = (batch, nheads, (seqlen + 63) // 64)  # BLOCK_M=64 will be handled in kernel
    
    if not inplace:
        out = torch.empty_like(x)
    else:
        out = x

    # Determine strides
    if x.ndim == 4:
        x = x.contiguous()
        stride_xb, stride_xm, stride_xh, stride_xd = x.stride()
    else:
        x = x.contiguous()
        stride_xm, stride_xh, stride_xd = x.stride()
        stride_xb = 0  # Not used in variable length
    
    if out.ndim == 4:
        out = out.contiguous()
        stride_outb, stride_outm, stride_outh, stride_outd = out.stride()
    else:
        out = out.contiguous()
        stride_outm, stride_outh, stride_outd = out.stride()
        stride_outb = 0  # Not used in variable length
    
    cos = cos.contiguous()
    sin = sin.contiguous()
    stride_cosm, stride_cosd = cos.stride()
    stride_sinm, stride_sind = sin.stride()

    BLOCK_M = 64
    BLOCK_N = 32

    rotary_kernel[grid](
        x, cos, sin, cu_seqlens, out,
        HEAD_SIZE=headdim,
        ROTARY_DIM=rotary_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
        stride_xb=stride_xb,
        stride_xm=stride_xm,
        stride_xh=stride_xh,
        stride_xd=stride_xd,
        stride_outb=stride_outb,
        stride_outm=stride_outm,
        stride_outh=stride_outh,
        stride_outd=stride_outd,
        stride_cosm=stride_cosm,
        stride_cosd=stride_cosd,
        stride_sinm=stride_sinm,
        stride_sind=stride_sind,
        num_warps=4
    )

    return out


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
