
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'SPLIT_K': 2}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = first_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn)

    scales_ptrs = scales_ptr + (offs_n * stride_scales_n)
    zeros_ptrs = zeros_ptr + ((offs_n // 8) * stride_zeros_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        mask_k = offs_k < K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)

        b_idx = (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
        b_raw = tl.load(b_ptr + b_idx, mask=mask_k[:, None], other=0)

        b_i4 = (b_raw >> (4 * (offs_k[:, None] % 8))) & 0xF

        group_idx = (k * BLOCK_SIZE_K * SPLIT_K + offs_k[:, None]) // groupsize
        scales = tl.load(scales_ptrs + group_idx * stride_scales_g)
        zeros = tl.load(zeros_ptrs + group_idx * stride_zeros_g)

        b_fp = (b_i4 - ((zeros >> (4 * (offs_n[None, :] % 8))) & 0xF)) * scales

        accumulator += tl.dot(a, b_fp.to(a.dtype))

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk
        offs_k += BLOCK_SIZE_K * SPLIT_K

    offs_m_real = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_real = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m_real < M
    mask_n = offs_n_real < N

    c_ptrs = c_ptr + stride_cm * offs_m_real[:, None] + stride_cn * offs_n_real[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]

    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


def matmul_dequantize_int4_s2(
    x: torch.FloatTensor,
    qweight: torch.IntTensor,
    scales: torch.FloatTensor,
    qzeros: torch.IntTensor,
    groupsize: int = 128,
    output=None
) -> torch.FloatTensor:
    assert x.is_contiguous(), "x must be contiguous"
    assert qweight.is_contiguous(), "qweight must be contiguous"

    M, K = x.shape
    assert K == qweight.shape[0] * 8, "K must align with packed INT4 weight size"
    N = qweight.shape[1]

    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META.get('SPLIT_K', 1),
    )

    matmul_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        groupsize,
        GROUP_SIZE_M=8,
    )
    return output


def quantize_int4(w: torch.Tensor, groupsize: int = 128):
    assert w.dtype in (torch.float16, torch.float32, torch.bfloat16)
    w = w.float()

    N, K = w.shape
    assert K % groupsize == 0, "groupsize must evenly divide K"

    w_grouped = w.view(N, K // groupsize, groupsize)
    mn, _ = w_grouped.min(dim=-1, keepdim=True)
    mx, _ = w_grouped.max(dim=-1, keepdim=True)
    scales = (mx - mn) / 15.0
    zeros = -mn / scales

    quantized = torch.round((w_grouped - mn) / scales).clamp(0, 15).to(torch.int8)

    quantized = quantized.view(N, K)
    scales = scales.view(N, K // groupsize)
    zeros = zeros.to(torch.int8).view(N, K // groupsize)

    packed = torch.zeros((N, K // 2), dtype=torch.int32, device=w.device)
    for k in range(0, K, 2):
        even = quantized[:, k]
        odd = quantized[:, k + 1] if k + 1 < K else 0
        packed[:, k // 2] = (odd.int() << 4) | (even.int() & 0xF)

    return packed, scales.float(), zeros


def unpack_int4(b_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, groupsize: int = 128):
    assert b_packed.dtype == torch.int32
    assert b_packed.dim() == 2
    N, K_half = b_packed.shape
    K = K_half * 2

    unpacked = torch.empty((N, K), dtype=torch.int8, device=b_packed.device)
    for k in range(0, K // 2):
        val = b_packed[:, k]
        unpacked[:, 2 * k] = val.int() & 0xF
        unpacked[:, 2 * k + 1] = (val.int() >> 4) & 0xF

    group_idx = torch.arange(K, device=b_packed.device) // groupsize
    zeros_exp = zeros.gather(1, group_idx.view(1, -1).expand(N, K))
    scales_exp = scales.gather(1, group_idx.view(1, -1).expand(N, K))

    return (unpacked.float() - zeros_exp) * scales_exp

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
