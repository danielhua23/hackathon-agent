
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  stride_scales_g, stride_scales_n,
                  stride_zeros_g, stride_zeros_n,
                  M, N, K,
                  groupsize,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  SPLIT_K: tl.constexpr,
                  IS_EVEN_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m * BLOCK_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_am = offs_m % M
    offs_bn = offs_n % N
    offs_bk = offs_k % K

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_bk[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_bk[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + ((offs_bn[None, :] // groupsize) * stride_scales_g + offs_bn[None, :] * stride_scales_n)
    zeros_ptrs = zeros_ptr + ((offs_bn[None, :] // groupsize) * stride_zeros_g + offs_bn[None, :] * stride_zeros_n)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        mask_k = IS_EVEN_K or (offs_bk[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_i4 = tl.load(b_ptrs, mask=mask_k, other=0)
        scales = tl.load(scales_ptrs)
        zeros = tl.load(zeros_ptrs)

        b_i4 = (b_i4 >> ((offs_bk[:, None] % 8) * 4)) & 0xF
        b = b_i4.to(tl.float32) * scales - zeros
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask)


configs = [
    triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1}, num_stages=2,
                  num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1}, num_stages=2,
                  num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1}, num_stages=2,
                  num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 1}, num_stages=2,
                  num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'SPLIT_K': 2}, num_stages=2,
                  num_warps=4),
]


@triton.autotune(configs=configs, key=['M', 'N', 'K'])
@triton.jit
def matmul_dequantize_int4_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                                  stride_am, stride_ak,
                                  stride_bk, stride_bn,
                                  stride_cm, stride_cn,
                                  stride_scales_g, stride_scales_n,
                                  stride_zeros_g, stride_zeros_n,
                                  M, N, K,
                                  groupsize,
                                  BLOCK_SIZE_M: tl.constexpr,
                                  BLOCK_SIZE_N: tl.constexpr,
                                  BLOCK_SIZE_K: tl.constexpr,
                                  SPLIT_K: tl.constexpr):
    matmul_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  stride_scales_g, stride_scales_n,
                  stride_zeros_g, stride_zeros_n,
                  M, N, K,
                  groupsize,
                  BLOCK_SIZE_M=BLOCK_SIZE_M,
                  BLOCK_SIZE_N=BLOCK_SIZE_N,
                  BLOCK_SIZE_K=BLOCK_SIZE_K,
                  SPLIT_K=SPLIT_K,
                  IS_EVEN_K=(K % (BLOCK_SIZE_K * SPLIT_K) == 0))


def matmul_dequantize_int4_s2(a: torch.Tensor, b: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                              groupsize: int):
    assert a.dtype == torch.float16 or a.dtype == torch.float32
    assert b.dtype == torch.int8
    assert scales.dtype == torch.float16 or scales.dtype == torch.float32
    assert zeros.dtype == torch.float16 or zeros.dtype == torch.float32

    M, K = a.shape
    K_, N = b.shape
    assert K * 8 // 4 == K_, "Weight shape mismatch (K in int4)"
    assert scales.shape == zeros.shape == (N, K // groupsize)

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    grid_lambda = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META['SPLIT_K'])

    matmul_dequantize_int4_kernel[grid_lambda](
        a, b, c, scales, zeros,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0) if scales.dim() > 1 else 0, scales.stride(1),
        zeros.stride(0) if zeros.dim() > 1 else 0, zeros.stride(1),
        M, N, K,
        groupsize,
    )

    return c


def quantize_int4(w: torch.Tensor, groupsize: int = 128):
    assert w.dim() == 2, "Weight tensor must be 2-D"
    w = w.to(torch.float32)
    oc, ic = w.shape
    assert ic % groupsize == 0

    w = w.reshape(oc, ic // groupsize, groupsize)
    wmax = w.amax(dim=2, keepdim=True)
    wmin = w.amin(dim=2, keepdim=True)

    scale = (wmax - wmin) / 15
    zero = -wmin / scale
    scale = scale.squeeze(-1)
    zero = zero.squeeze(-1)

    int_w = torch.round((w - wmin) / scale.unsqueeze(-1)).clamp(0, 15)
    int_w = int_w.to(torch.int8)

    out = torch.zeros(oc, ic // 8, dtype=torch.int32, device=w.device)
    for i in range(0, ic, 8):
        out_i = 0
        for j in range(8):
            out_i |= (int_w[:, i//groupsize, i%groupsize + j] << (j * 4))
        out[:, i//8] = out_i

    scale = scale.to(torch.float16)
    zero = zero.to(torch.float16)

    return out.reshape(oc, -1), scale, zero


def unpack_int4(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, groupsize: int = 128):
    assert w.dim() == 2
    oc, ic_int = w.shape
    ic = ic_int * 8
    assert ic % groupsize == 0

    w_bits = torch.empty(oc, ic, dtype=torch.int8, device=w.device)
    for i in range(ic):
        shift = (i % 8) * 4
        w_bits[:, i] = (w[:, i // 8] >> shift) & 0xF

    scale = scale.unsqueeze(-1).expand_as(w_bits)
    zero = zero.unsqueeze(-1).expand_as(w_bits)
    w_bits = w_bits.to(scale.dtype)

    w_unpacked = scale * w_bits - zero
    return w_unpacked.to(torch.float16)


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
