
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
    stride_x_batch,
    stride_x_head,
    stride_x_m,
    stride_x_k,
    stride_c_stride,
    stride_cos_m,
    stride_cos_k,
    stride_sin_m,
    stride_sin_k,
    stride_out_batch,
    stride_out_head,
    stride_out_m,
    stride_out_k,
    n_ctx,
    HEAD_K: tl.constexpr,
    IS_VARIABLE_KV: tl.constexpr,
    CONJUGATE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    pid_m = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_h = pid_head < HEAD_K
    mask_m = pid_m < n_ctx

    if IS_VARIABLE_KV:
        # Handle variable sequence lengths
        cu_seq = tl.load(CU_SEQLENS + pid_batch)
        seq_len = tl.load(SEQLENS + pid_batch)
        offset_m = cu_seq + pid_m
    else:
        # Handle fixed sequence length
        offset_m = pid_batch * n_ctx + pid_m
        seq_len = n_ctx

    mask_seq = pid_m < seq_len

    if INTERLEAVED:
        # Interleaved format: real and imag parts are interleaved
        load_real_idx = 2 * pid_m + 0
        load_imag_idx = 2 * pid_m + 1

        off_real = (
            pid_batch * stride_x_batch
            + pid_head[None, :] * stride_x_head
            + load_real_idx[:, None] * stride_x_m
            + tl.arange(0, HEAD_K // 2)[None, :] * stride_x_k
        )
        off_imag = (
            pid_batch * stride_x_batch
            + pid_head[None, :] * stride_x_head
            + load_imag_idx[:, None] * stride_x_m
            + tl.arange(0, HEAD_K // 2)[None, :] * stride_x_k
        )

        # Load real and imaginary parts
        x_real = tl.load(X + off_real, mask=mask_m[:, None] & mask_h[None, :], other=0.0)
        x_imag = tl.load(X + off_imag, mask=mask_m[:, None] & mask_h[None, :], other=0.0)

        # Load COS and SIN
        off_cos_m = offset_m[:, None] * stride_cos_m
        off_sin_m = offset_m[:, None] * stride_sin_m

        # Get the right dimension for COS/SIN
        off_cos_real = (
            off_cos_m
            + (2 * tl.arange(0, HEAD_K // 2))[None, :] * stride_cos_k
        )
        off_sin_real = (
            off_sin_m
            + (2 * tl.arange(0, HEAD_K // 2))[None, :] * stride_sin_k
        )
        off_cos_imag = (
            off_cos_m
            + (2 * tl.arange(0, HEAD_K // 2) + 1)[None, :] * stride_cos_k
        )
        off_sin_imag = (
            off_sin_m
            + (2 * tl.arange(0, HEAD_K // 2) + 1)[None, :] * stride_sin_k
        )

        cos_real = tl.load(COS + off_cos_real, mask=mask_m[:, None], other=1.0)
        sin_real = tl.load(SIN + off_sin_real, mask=mask_m[:, None], other=0.0)
        cos_imag = tl.load(COS + off_cos_imag, mask=mask_m[:, None], other=0.0)
        sin_imag = tl.load(SIN + off_sin_imag, mask=mask_m[:, None], other=0.0)

    else:
        # Non-interleaved format: first half is real, second half is imag
        half_k = HEAD_K // 2

        # Offsets for real and imaginary parts
        off_real = (
            pid_batch * stride_x_batch
            + pid_head[None, :] * stride_x_head
            + pid_m[:, None] * stride_x_m
            + tl.arange(0, half_k)[None, :] * stride_x_k
        )
        off_imag = (
            pid_batch * stride_x_batch
            + (half_k + pid_head)[None, :] * stride_x_head
            + pid_m[:, None] * stride_x_m
            + tl.arange(0, half_k)[None, :] * stride_x_k
        )

        # Load real and imaginary parts
        x_real = tl.load(X + off_real, mask=mask_m[:, None] & (pid_head < half_k)[None, :], other=0.0)
        x_imag = tl.load(X + off_imag, mask=mask_m[:, None] & (pid_head >= half_k)[None, :], other=0.0)

        # Load COS and SIN for non-interleaved
        off_cos = (
            offset_m[:, None] * stride_cos_m
            + tl.arange(0, half_k)[None, :] * stride_cos_k
        )
        off_sin = (
            offset_m[:, None] * stride_sin_m
            + tl.arange(0, half_k)[None, :] * stride_sin_k
        )

        cos = tl.load(COS + off_cos, mask=mask_m[:, None], other=1.0)
        sin = tl.load(SIN + off_sin, mask=mask_m[:, None], other=0.0)

        cos_real = cos
        sin_real = sin
        cos_imag = cos
        sin_imag = sin

    # Compute rotary transform
    if CONJUGATE:
        # With conjugation
        out_real = x_real * cos_real + x_imag * sin_real
        out_imag = -x_real * sin_imag + x_imag * cos_imag
    else:
        # Without conjugation
        out_real = x_real * cos_real - x_imag * sin_real
        out_imag = x_real * sin_imag + x_imag * cos_imag

    # Store results
    if INTERLEAVED:
        off_out_real = (
            pid_batch * stride_out_batch
            + pid_head[None, :] * stride_out_head
            + load_real_idx[:, None] * stride_out_m
            + tl.arange(0, HEAD_K // 2)[None, :] * stride_out_k
        )
        off_out_imag = (
            pid_batch * stride_out_batch
            + pid_head[None, :] * stride_out_head
            + load_imag_idx[:, None] * stride_out_m
            + tl.arange(0, HEAD_K // 2)[None, :] * stride_out_k
        )
        tl.store(OUT + off_out_real, out_real, mask=mask_m[:, None] & mask_h[None, :])
        tl.store(OUT + off_out_imag, out_imag, mask=mask_m[:, None] & mask_h[None, :])
    else:
        off_out_real = (
            pid_batch * stride_out_batch
            + pid_head[None, :] * stride_out_head
            + pid_m[:, None] * stride_out_m
            + tl.arange(0, half_k)[None, :] * stride_out_k
        )
        off_out_imag = (
            pid_batch * stride_out_batch
            + (half_k + pid_head)[None, :] * stride_out_head
            + pid_m[:, None] * stride_out_m
            + tl.arange(0, half_k)[None, :] * stride_out_k
        )
        tl.store(OUT + off_out_real, out_real, mask=mask_m[:, None] & (pid_head < half_k)[None, :])
        tl.store(OUT + off_out_imag, out_imag, mask=mask_m[:, None] & (pid_head >= half_k)[None, :])


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offset: int = 0,
    cu_seqlens: torch.Tensor = None,
    seqlens: torch.Tensor = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    # Determine output tensor
    out = x if inplace else torch.empty_like(x)

    # Get dimensions
    batch, head, n_ctx, head_k = x.shape

    # Determine if variable sequence lengths are used
    is_variable_kv = cu_seqlens is not None and seqlens is not None

    # Building the grid
    BLOCK_H = 64
    BLOCK_M = 32
    grid = (batch, triton.cdiv(head, BLOCK_H), triton.cdiv(n_ctx, BLOCK_M))

    # Launch the kernel
    rotary_kernel[grid](
        x,
        cos,
        sin,
        out,
        cu_seqlens,
        seqlens,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        1 if is_variable_kv else 0,
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        n_ctx,
        HEAD_K=head_k,
        IS_VARIABLE_KV=is_variable_kv,
        CONJUGATE=conjugate,
        INTERLEAVED=interleaved,
        BLOCK_H=BLOCK_H,
        BLOCK_M=BLOCK_M,
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
