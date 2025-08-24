
import torch
import triton
import triton.language as tl

# int4 de-quant helpers
@triton.jit
def _dequantize_int4_unpack(xi32, mask0=0x0f, mask1=0xf0):
    xi0 = (xi32 & mask0).to(tl.int8)
    xi1 = ((xi32 & mask1) >> 4).to(tl.int8)
    return xi0, xi1


@triton.jit
def _dequantize_int4_kernel(ptr, scales_ptr, zeros_ptr, M, N,
                            stride_q, stride_s, stride_z,
                            BLOCK_M: tl.constexpr,
                            BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    q_offsets = (rm[:, None] * stride_q + (rn // 8)[None, :])
    scales_offsets = (rm[:, None] * stride_s + (rn // 8)[None, :])
    zeros_offsets = (rm[:, None] * stride_z + (rn // 8)[None, :])

    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m[:, None] & mask_n[None, :]

    packed = tl.load(ptr + q_offsets, mask=mask, other=0)
    s = tl.load(scales_ptr + scales_offsets, mask=mask, other=1.0)
    z = tl.load(zeros_ptr + zeros_offsets, mask=mask, other=0.0)

    offsets_0 = (rn % 8) * 4
    offsets_1 = offsets_0 + 4
    i0, i1 = _dequantize_int4_unpack(packed)
    v0 = (i0.to(tl.float32) - z) * s
    v1 = (i1.to(tl.float32) - z) * s

    return v0, v1


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  scales_ptr, zeros_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  stride_eval_k, stride_eval_n,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr,
                  SPLIT_K: tl.constexpr,
                  EVEN_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    n_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    n_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        group_id = 0
        first_pid_m = 0
    else:
        group_id = pid_m // GROUP_SIZE_M
        first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(n_blocks_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m % group_size_m)

    if SPLIT_K > 1:
        local_k = tl.cdiv(K, SPLIT_K)
        k_offset = pid_k * local_k
    else:
        local_k = K
        k_offset = 0

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn

    scales_ptrs = scales_ptr + ((offs_k[:, None] // 8) * stride_eval_k) + offs_n[None, :] * stride_eval_n
    zeros_ptrs = zeros_ptr + ((offs_k[:, None] // 8) * stride_eval_k) + offs_n[None, :] * stride_eval_n

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, local_k, BLOCK_SIZE_K):
        if EVEN_K or (k + BLOCK_SIZE_K <= local_k):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < local_k - k, other=0.0, eviction_policy="evict_last")
            block_scale = tl.load(scales_ptrs, mask=offs_k[:, None] < local_k - k, other=1.0)
            block_zero = tl.load(zeros_ptrs, mask=offs_k[:, None] < local_k - k, other=0.0)

            packed_b = tl.load(b_ptrs, mask=offs_k[:, None] < local_k - k, other=0)
            k_idx = (offs_k[:, None] % 8) * 4
            val_low = (packed_b & 0x0F).to(tl.int8).to(tl.float32)
            val_high = ((packed_b >> 4) & 0x0F).to(tl.int8).to(tl.float32)
            b_low = (val_low - block_zero) * block_scale
            b_high = (val_high - block_zero) * block_scale

            acc = tl.dot(a, b_low, acc)
            a_shift = tl.load(a_ptrs + stride_bk * (1 if EVEN_K else 8), mask=offs_k[None, :] + 8 < local_k - k, other=0.0, eviction_policy="evict_last")
            acc = tl.dot(a_shift, b_high, acc)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk
        scales_ptrs += (BLOCK_SIZE_K // 8) * stride_eval_k
        zeros_ptrs += (BLOCK_SIZE_K // 8) * stride_eval_k

    if SPLIT_K == 1:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptrs.type.element_ty), mask=c_mask)
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + pid_k * M * N
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


def matmul_dequantize_int4_s2(a, int4b_compressed, scales, zeros, M, N, K):
    c_dtype = a.dtype
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    SPLIT_K = 1

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),
                         triton.cdiv(N, META['BLOCK_SIZE_N']),
                         SPLIT_K)

    if SPLIT_K > 1:
        c = torch.empty((SPLIT_K, M, N), dtype=torch.float32, device=a.device)
    else:
        c = torch.empty((M, N), dtype=c_dtype, device=a.device)

    EVEN_K = K % 32 == 0

    matmul_kernel[grid](a, int4b_compressed, c,
                        scales, zeros,
                        M, N, K,
                        a.stride(0), a.stride(1),
                        int4b_compressed.stride(0), int4b_compressed.stride(1),
                        c.stride(0) if c.dim() == 2 else c.stride(1),
                        c.stride(1) if c.dim() == 2 else c.stride(2),
                        scales.stride(0), scales.stride(1),
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        GROUP_SIZE_M=8,
                        SPLIT_K=SPLIT_K,
                        EVEN_K=EVEN_K)
    return c if SPLIT_K == 1 else c.sum(dim=0)


def quantize_int4(x: torch.Tensor, group_size: int = 128):
    org_shape = x.shape
    x = x.view(-1, group_size)
    x_fp32 = x.float()
    x_min = x_fp32.amin(dim=-1, keepdim=True)
    x_max = x_fp32.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min) / 15.0
    zero = (-x_min / scale + 0.5).clamp(0, 15)

    int4 = (((x_fp32 - x_min) / scale + 0.5).clamp(0, 15)).round().to(torch.int8)

    int4 = int4.view(-1)
    packed = torch.zeros(x.size(0), x.size(1) // 2, dtype=torch.int32, device=x.device)
    int4_even = int4[::2].to(torch.int32)
    int4_odd = int4[1::2].to(torch.int32)
    packed = int4_even | (int4_odd << 4)
    packed = packed.view(org_shape[0], org_shape[1] // 2)
    scale = scale.squeeze(-1)
    zero = zero.squeeze(-1)
    return packed, scale, zero


def unpack_int4(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group_size: int = 128):
    M, Nhalf = packed.shape
    N = Nhalf * 2
    unpacked = torch.empty(M, N, dtype=torch.float16, device=packed.device)

    packed = packed.int()
    for i in range(M):
        for j in range(Nhalf):
            low = (packed[i, j] & 0x0F).to(torch.float32)
            high = ((packed[i, j] >> 4) & 0x0F).to(torch.float32)
            group_idx = j * 2 // group_size
            val_low = (low - zero[i, group_idx]) * scale[i, group_idx]
            val_high = (high - zero[i, group_idx]) * scale[i, group_idx]
            unpacked[i, 2 * j] = val_low.to(torch.float16)
            unpacked[i, 2 * j + 1] = val_high.to(torch.float16)
    return unpacked


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
