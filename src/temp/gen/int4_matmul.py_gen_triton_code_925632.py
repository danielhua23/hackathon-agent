
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'NO_GROUPS'],
)
@triton.jit
def gptq_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize, NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    bits = 4
    infearure_per_bits = 8
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs  = zeros_ptr  + ((offs_bn // infearure_per_bits) * stride_zeros_n)

    shifter    = ((offs_k % infearure_per_bits) * bits)[:, None]
    zeros_shift = ((offs_bn % infearure_per_bits) * bits)[None, :]

    if NO_GROUPS:
        scales = tl.load(scales_ptrs)
        zeros  = tl.load(zeros_ptrs)
        zeros_int = (zeros >> zeros_shift) & 0xF
        zeros = zeros_int * scales
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b_i32 = tl.load(b_ptrs)
        b_u8  = (b_i32 >> shifter) & 0xF
        b_fp  = b_u8.to(tl.float32)

        if not NO_GROUPS:
            g_id = k // (groupsize // BLOCK_SIZE_K)
            ptr_s = scales_ptrs + g_id * stride_scales_g
            ptr_z = zeros_ptrs  + g_id * stride_zeros_g
            scales = tl.load(ptr_s)
            zeros  = tl.load(ptr_z)
            zeros_int = (zeros >> zeros_shift) & 0xF
            zeros = zeros_int * scales

        b = b_fp * scales - zeros
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=mask)


def matmul_dequantize_int4_gptq(x: torch.FloatTensor,
                                 qweight: torch.IntTensor,
                                 scales: torch.FloatTensor,
                                 qzeros: torch.IntTensor,
                                 group_size) -> torch.FloatTensor:
    assert x.dim() == 2 and qweight.dim() == 2
    assert x.shape[-1] == (qweight.shape[0] * 8), "x inner dim mismatch"
    assert x.is_contiguous(), "x must be contiguous"

    M, K = x.shape
    N = qweight.shape[1]
    output = torch.empty((M, N), device=x.device, dtype=torch.float16)

    def grid(META): return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    gptq_gemm_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0),  output.stride(1),
        scales.stride(0)  if scales.dim() > 1 else 0,
        scales.stride(1),
        qzeros.stride(0)  if qzeros.dim() > 1 else 0,
        qzeros.stride(1),
        group_size, group_size == K,
    )
    return output


configs_s2 = [
    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128,'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
]


@triton.autotune(configs=configs_s2, key=['M', 'N', 'K'])
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
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    GROUP_SIZE_M_local = 8
    num_pid_in_group = GROUP_SIZE_M_local * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M_local
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M_local)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = (pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_cur = offs_k[None, :] + k * BLOCK_SIZE_K * SPLIT_K
        valid_k = k_cur < K

        scale_ptrs = scales_ptr + (k_cur // groupsize) * stride_scales_g + offs_n[None, :] * stride_scales_n
        zeros_ptrs = zeros_ptr  + (k_cur // groupsize) * stride_zeros_g  + (offs_n[None, :] // 8) * stride_zeros_n

        a = tl.load(a_ptrs, mask=valid_k, other=0.0)
        b_i32 = tl.load(b_ptrs, mask=valid_k, other=0)

        scales = tl.load(scale_ptrs, mask=valid_k, other=0.0)
        zeros  = tl.load(zeros_ptrs, mask=valid_k, other=0)

        b_shift = (k_cur % 8) * 4
        zeros_shift = ((offs_n[None, :] % 8) * 4)
        b_i4 = (b_i32 >> b_shift) & 0xF
        zp_i4 = (zeros >> zeros_shift) & 0xF
        b_fp = (b_i4 - zp_i4).to(tl.float16) * scales.to(tl.float16)

        acc += tl.dot(a.to(tl.float16), b_fp)

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask)


def matmul_dequantize_int4_s2(x: torch.FloatTensor,
                              qweight: torch.IntTensor,
                              scales: torch.FloatTensor,
                              qzeros: torch.IntTensor,
                              groupsize: int = 128) -> torch.FloatTensor:
    assert x.is_contiguous() and qweight.is_contiguous()
    M, K = x.shape
    K_, N = qweight.shape
    assert K * 8 // 4 == K_, "K dim mismatch"
    assert scales.shape == zeros.shape == (N, K // groupsize)

    c = torch.empty((M, N), device=x.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META['SPLIT_K'])
    matmul_dequantize_int4_kernel[grid](
        x, qweight, c, scales, qzeros,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0),  c.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        M, N, K,
        groupsize,
    )
    return c


def quantize_int4(w: torch.Tensor, groupsize: int = 128):
    assert w.dim() == 2
    w = w.float()
    oc, ic = w.shape
    assert ic % groupsize == 0

    w = w.reshape(oc, ic // groupsize, groupsize)
    wmax = w.amax(dim=2, keepdim=True)
    wmin = w.amin(dim=2, keepdim=True)
    scale = (wmax - wmin) / 15
    zero = (-wmin / scale).round().clamp(0, 15)
    scale = scale.squeeze(-1)
    zero  = zero.squeeze(-1)

    int_w = torch.round((w - wmin) / scale.unsqueeze(-1)).clamp(0, 15).to(torch.int8)

    out = torch.zeros(oc, ic // 8, dtype=torch.int32, device=w.device)
    for i in range(0, ic, 8):
        packed = 0
        for j in range(8):
            packed |= int_w[:, i // groupsize, i % groupsize + j] << (j * 4)
        out[:, i // 8] = packed

    return out.reshape(oc, -1), scale.half(), zero.half()


def unpack_int4(w_packed: torch.IntTensor, scale: torch.Tensor, zero: torch.Tensor, groupsize: int = 128):
    oc, ic_int = w_packed.shape
    ic = ic_int * 8
    w_bits = torch.empty(oc, ic, dtype=torch.float32, device=w_packed.device)
    for i in range(ic):
        shift = (i % 8) * 4
        w_bits[:, i] = torch.bitwise_and(torch.bitwise_right_shift(w_packed[:, i // 8], shift), 0xF).float()

    scale = scale.unsqueeze(-1).expand_as(w_bits)
    zero  = zero.unsqueeze(-1).expand_as(w_bits)
    return (scale * w_bits - zero).half()


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
