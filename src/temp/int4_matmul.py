
import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------------
# Triton kernels for INT4 matrix multiplication (weight dequantized on the fly)
# --------------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # M,   N,   K,  BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_stages, num_warps
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
    ],
    key = ["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # pointers to matmul operands
    a_ptr, b_ptr, c_ptr,                          # a is fp16/bf16, b is quantized (int packed), c is output fp16/bf16
    # scales + zero points vectors
    scales_ptr, zeros_ptr,                        # per-group fp16
    # strides
    stride_am, stride_ak,
    stride_bk, stride_bn, stride_b_packed,        # b is (K/8, N)  packed 8 int4 in one int32
    stride_cm, stride_cn,
    stride_scales,                                # (num_groups)
    stride_zeros,                                 # (num_groups)
    # dimension sizes
    M, N, K,
    groupsize: tl.constexpr,                      # dequantization group granularity
    # block sizes for tiling
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid  = tl.program_id(axis=0)
    pid_z= tl.program_id(axis=1)                 # for SPLIT_K

    # tile identifiers
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if SPLIT_K > 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_tiles_k = tl.cdiv(K, BLOCK_SIZE_K)
        pid_m = pid // (num_tiles_k * num_pid_n)
        remaining = pid % (num_tiles_k * num_pid_n)
        pid_n = remaining // num_tiles_k
        pid_k_first = remaining % num_tiles_k
        pid_k_last = pid_k_first + 1
    # NOTE: currently implement simple row/col tiling, so we set SPLIT_K always to 1
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # offset block pointers
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # adjust overlapping
    offs_m = tl.where(offs_m < M, offs_m, M-1)
    offs_n = tl.where(offs_n < N, offs_n, N-1)
    offs_k = tl.where(offs_k < K, offs_k, K-1)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_b_packed + offs_n[None, :] * stride_bn)

    scales_ptrs = scales_ptr + ((offs_k[:, None] // groupsize) * stride_scales)
    zeros_ptrs  = zeros_ptr  + ((offs_k[:, None] // groupsize) * stride_zeros)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # edge masking
        k_cur = k * BLOCK_SIZE_K + offs_k
        mask_k = k_cur < K

        # load A tile (fp16)
        a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)

        # load packed INT4 B tile
        b_int = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0)

        # ---- dequantize ----
        # unpack each int32 into 8 int4 values (low nibble first)
        scales = tl.load(scales_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=1.0)
        zeros  = tl.load(zeros_ptrs,  mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)

        # split nibble from packed int8
        inner = (offs_k[:, None] % 8) * 4
        b_ext   = (b_int >> inner) & 0xF          # 0..15
        b_deint = b_ext.to(tl.float32)

        bq_f32 = scales * (b_deint - zeros)

        # emulated block-K reduction accumulation
        accumulator += tl.dot(a_tile, bq_f32)

        # advance pointers
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_b_packed
        scales_ptrs += (BLOCK_SIZE_K * SPLIT_K // groupsize) * stride_scales
        zeros_ptrs  += (BLOCK_SIZE_K * SPLIT_K // groupsize) * stride_zeros

    # write back
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_cm < M
    mask_n = offs_cn < N
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c = accumulator.to(c_ptr.type.element_ty)
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])



# --------------------------------------------------------------------------------
# Python utility entry — int4 dequantized matrix multiply wrapper
# --------------------------------------------------------------------------------
def matmul_dequantize_int4_s2(
    x: torch.Tensor,                   # (M, K)  fp16/fp32
    qweight: torch.Tensor,             # (K//8, N) int32 each value holds 8 int4
    scales: torch.Tensor,              # (num_groups, N) fp16/fp32
    zeros: torch.Tensor,               # (num_groups, N) fp16/fp32
    groupsize: int = 128,
) -> torch.Tensor:
    # Device check (ROCm friendly)
    assert x.is_cuda or str(x.device).startswith("cuda")
    M, K = x.shape
    assert qweight.shape == (K//8, qweight.shape[1])
    N = qweight.shape[1]

    # alloc output
    c = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # prepare grid
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), 1)

    matmul_kernel[grid](
        x, qweight, c,
        scales, zeros,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1), qweight.stride(0),  # 3rd stride unused in kernel
        c.stride(0), c.stride(1),
        scales.stride(0), zeros.stride(0),
        M, N, K,
        groupsize,
    )

    return c



# --------------------------------------------------------------------------------
# INT4 quantize helper
# --------------------------------------------------------------------------------
def quantize_int4(w: torch.Tensor, groupsize: int = 128) -> tuple:
    """
    Quantize fp16/32 weights into INT4 with per-group scale & zero-point.
    Returns:
        qw      (K//8, N)  int32   -> 8 int4 per int32
        scales  (num_groups, N) fp16
        zeros   (num_groups, N) fp16
    """
    if w.dim() == 1:
        w = w.unsqueeze(1)
    shape = w.shape
    K_orig, N = shape[-2], shape[-1]
    w = w.view(-1, N)

    # pad to multiple of groupsize
    K_pad = (K_orig + groupsize - 1) // groupsize * groupsize
    if K_pad > K_orig:
        w = torch.cat([w, torch.zeros(K_pad - K_orig, N, dtype=w.dtype, device=w.device)], dim=0)

    assert w.shape[0] % groupsize == 0
    num_groups = w.shape[0] // groupsize

    # Reshape to (num_groups, groupsize, N)
    w = w.view(num_groups, groupsize, N)

    # compute scale & zero
    w_min = torch.amin(w, dim=1)  # (num_groups,N)
    w_max = torch.amax(w, dim=1)
    scale = (w_max - w_min) / 15.0
    scale = scale.clamp(min=1e-10)
    zero = (torch.round(-w_min / scale)).clamp(0, 15)

    # quantize
    w_int = torch.round(w / scale.unsqueeze(1) + zero.unsqueeze(1)).clamp(0, 15).to(torch.int32)

    # pack 8 INT4 -> 1 INT32
    packed = torch.zeros(num_groups * groupsize // 8, N, dtype=torch.int32, device=w.device)
    for i in range(8):
        mask = 0xF
        packed |= (w_int[:, i::8, :] << (4 * i)) & mask

    packed = packed.view(K_pad // 8, N)
    scale = scale.to(torch.float16)
    zero  = zero.to(torch.float16)

    return packed[: (K_orig + 7) // 8], scale, zero


# --------------------------------------------------------------------------------
# Utility to unpack INT4 for testing only
# --------------------------------------------------------------------------------
@triton.jit
def _unpack_int4_kernel(
    qw_ptr, scales_ptr, zeros_ptr, out_ptr,
    K, N,
    stride_qw, stride_scales, stride_zeros, stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    # indices
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = (offs_k < K) & (offs_n < N)
    scale_ptr = scales_ptr + offs_n * stride_scales
    zero_ptr  = zeros_ptr  + offs_n * stride_zeros
    scales = tl.load(scale_ptr, mask=offs_n < N, other=1.0)
    zeros  = tl.load(zero_ptr , mask=offs_n < N, other=0.0)

    # Each qw elt holds 8 values
    offs_k_group = offs_k // 8
    offs_k_inner = offs_k % 8

    qw_idx = offs_k_group * stride_qw + offs_n * 1  # contig along N
    qw = tl.load(qw_ptr + qw_idx, mask=mask, other=0)

    val = (qw >> (4 * offs_k_inner)) & 0xF
    fp_val = scales * (val.to(tl.float32) - zeros)
    offs_out = offs_k * stride_out + offs_n * 1
    tl.store(out_ptr + offs_out, fp_val, mask=mask)


def unpack_int4(qw: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    K8, N = qw.shape
    K = K8 * 8
    assert scales.shape == zeros.shape == (K // 128, N)  # depends on groupsize 128
    out = torch.zeros(K, N, dtype=scales.dtype, device=qw.device)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), triton.cdiv(K, META['BLOCK_SIZE']))

    _unpack_int4_kernel[grid](
        qw,
        scales,
        zeros,
        out,
        K, N,
        qw.stride(0), scales.stride(0), zeros.stride(0), out.stride(0),
        BLOCK_SIZE=64,
    )

    return out

##################################################################################################################################################



def test_correct_int4_s2(M=32, K=4096, N=4096):
    group_size = 128
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    int_b, b_scale, b_zero_point, _ = quantize_int4(b, group_size=group_size)
    
    # Test case
    triton_output = matmul_dequantize_int4_s2(a, int_b, b_scale, b_zero_point, group_size)
    
    results = {
        "test_case_1": triton_output
    }
    
    return results

result_gold = test_correct_int4_s2()
