

import torch
import triton
import triton.language as tl

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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Allocate shared memory for double buffering
    b_shared = tl.full([BLOCK_K, BLOCK_N], 0, dtype=tl.int32)
    b_next = tl.full([BLOCK_K, BLOCK_N], 0, dtype=tl.int32)
    b_buf = [b_shared, b_next]

    # Pre-load first tile
    k_start = 0
    k_offs = k_start + offs_k
    b = tl.load(b_ptrs)
    b_buf[0] = b

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Select current buffer (double buffering)
        b_tile = b_buf[k % 2]

        # Pre-load next tile in background if not last iteration
        if k + 1 < tl.cdiv(K, BLOCK_K):
            k_offs_next = (k + 1) * BLOCK_K + offs_k
            b_ptrs_next = b_ptr + ((k_offs_next[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn)
            b_next = tl.load(b_ptrs_next)
            b_buf[(k + 1) % 2] = b_next

        # Process current tile from shared memory
        k_offs = k * BLOCK_K + offs_k
        a_mask = k_offs[None, :] < K
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Unpack INT4 from shared memory
        b_int4 = ((b_tile >> ((k_offs[:, None] % 8) * 4)) & 0xF).to(tl.int32)

        group_idx = (k * BLOCK_K + offs_k) // GROUP_SIZE
        group_idx = group_idx[:, None]

        scales = tl.load(scales_ptr + group_idx * stride_scales_g + offs_n[None, :] * stride_scales_n)
        zeros = tl.load(zeros_ptr + group_idx * stride_zeros_g + offs_n[None, :] * stride_zeros_n)

        b_deq = (b_int4 - zeros) * scales

        acc += tl.dot(a, b_deq).to(tl.float32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 8) * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def quantize_int4(weights, group_size=128):
    """
    Quantize a float16 weight tensor to symmetric int4 with group-wise
    quantization.
    Each group has its own zero point and scale.

    Args:
        weights: a float tensor of shape [K, N]
        group_size: the group size

    Returns:
        int_weight: a int32 tensor of shape [K//8, N]
        scales: a float16 tensor of shape [num_groups, N]
        zeros: a float16 tensor of shape [num_groups, N]
    """
    K, N = weights.shape
    num_groups = K // group_size
    assert K % group_size == 0, "K must be divisible by group_size"

    weights = weights.view(num_groups, group_size, N)

    # Compute min, max in each group
    mins = weights.amin(dim=1)  # [num_groups, N]
    maxs = weights.amax(dim=1)  # [num_groups, N]

    scales = (maxs - mins) / 15.0
    zeros = torch.round(-mins / scales).clamp(0, 15).to(torch.int32)

    # Avoid division by zero
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    qw = torch.round(weights / scales[:, None, :] + zeros[:, None, :]).clamp(0, 15).to(torch.int32)

    # Pack int4 to int32 (8 values per int32)
    qw = qw.view(num_groups, group_size // 8, 8, N)
    int4_packed = torch.zeros(num_groups, group_size // 8, N, dtype=torch.int32, device=weights.device)
    for i in range(8):
        int4_packed |= qw[:, :, i, :] << (i * 4)

    # Flatten back to [K//8, N]
    int4_packed = int4_packed.view(K // 8, N)

    # Reshape scales and zeros
    scales = scales.to(torch.float16)
    zeros = zeros.to(torch.float16)

    return int4_packed, scales, zeros, None  # dummy arg returns


def unpack_int4(int_weight, scales, zeros, group_size):
    """
    De-quantize packed int4 weights back to float16.

    Args:
        int_weight: a int32 tensor of shape [K//8, N]
        scales: float16 tensor [num_groups, N]
        zeros: float16 tensor [num_groups, N]
        group_size: the group size used during quantization

    Returns:
        weights: a float16 tensor of shape [K, N]
    """
    K = int_weight.shape[0] * 8
    N = int_weight.shape[1]
    weights = torch.empty((K, N), dtype=torch.float16, device=int_weight.device)

    num_groups = K // group_size
    group_size_tiles = group_size // 8

    for g in range(num_groups):
        group_start = g * group_size
        int_group = int_weight[g * group_size_tiles: (g+1) * group_size_tiles]  # [group_size//8, N]
        
        unpacked = torch.empty((group_size, N), dtype=torch.float16, device=int_weight.device)
        
        for i in range(group_size//8):
            for j in range(8):
                val = (int_group[i] >> (j * 4)) & 0xF
                unpacked[i*8 + j] = (val.to(torch.float16) - zeros[g]) * scales[g]

        weights[group_start: group_start + group_size] = unpacked
    
    return weights


def matmul_dequantize_int4_s2(a, b, scales, zeros, group_size):
    """
    Perform matrix multiplication a @ b.T where b is quantized to int4.

    Args:
        a: float16 [M, K]
        b: int4-packed int32 [K//8, N]
        scales: float16 [num_groups, N] (num_groups = ceil_div(K, group_size))
        zeros: float16 [num_groups, N]
        group_size: the group size used in quantize_int4

    Returns:
        c: float16 [M, N]
    """
    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    kernel_config = {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE': group_size}

    matmul_kernel[grid](
        a, b, c,
        scales, zeros,
        M, N, 8 * K,  # K in rows of packed int4 is 8*external_K
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        **kernel_config
    )

    return c


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
