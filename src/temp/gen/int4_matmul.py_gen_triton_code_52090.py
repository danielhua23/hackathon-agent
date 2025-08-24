
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales, stride_zeros,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_pid_in_group = GROUP_SIZE_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(grid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 2) * stride_bk + offs_n[None, :] * stride_bn)

    scales_ptrs = scales_ptr + (offs_k * stride_scales)
    zeros_ptrs = zeros_ptr + (offs_k * stride_zeros)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_idx = k * BLOCK_SIZE_K * SPLIT_K + offs_k
        mask_k = k_idx[None, :] < K
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)

        b_idx = k_idx[:, None] // 2
        b_raw = tl.load(b_ptrs, mask=b_idx < (K * N) // 8, other=0)

        scales = tl.load(scales_ptrs, mask=k_idx < K, other=1.0)
        zeros = tl.load(zeros_ptrs, mask=k_idx < K, other=0.0)

        b_dequant = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(0, BLOCK_SIZE_K):
            sub_i = i // 2
            shift = (i % 2) * 4
            mask = tl.full((BLOCK_SIZE_N,), 0x0F, dtype=tl.int32)
            val = (b_raw[sub_i, :] >> shift) & mask
            val_f = val.to(tl.float32)
            dequant = val_f * scales[i] + zeros[i]
            b_dequant = tl.store(b_dequant, dequant, mask=i < BLOCK_SIZE_K)

        accumulator += tl.dot(a, b_dequant)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 2) * stride_bk
        scales_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_scales
        zeros_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_zeros

    if SPLIT_K > 1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=mask)
    else:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=mask)

def matmul_dequantize_int4_s2(a: torch.Tensor, b_quant: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, K: int) -> torch.Tensor:
    M, _ = a.shape
    _, N = b_quant.shape
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), 1)
    matmul_kernel[grid](
        a, b_quant, c,
        scales, zeros,
        M, N, K,
        a.stride(0), a.stride(1),
        b_quant.stride(0), b_quant.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), zeros.stride(0),
    )
    return c

def quantize_int4(weights: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_f = weights.to(torch.float32)
    shape = w_f.shape
    w_f = w_f.reshape(-1, group_size)
    w_min = w_f.min(dim=1, keepdim=True)[0]
    w_max = w_f.max(dim=1, keepdim=True)[0]
    scales = (w_max - w_min) / 15.0
    zeros = -w_min / scales
    w_int4 = torch.round((w_f - w_min) / scales).clamp(0, 15).to(torch.uint8)
    w_packed = torch.zeros(w_int4.shape[0], w_int4.shape[1] // 2, dtype=torch.int32, device=weights.device)
    for i in range(0, w_int4.shape[1], 2):
        val0 = w_int4[:, i].to(torch.int32)
        val1 = w_int4[:, i+1].to(torch.int32) if (i+1) < w_int4.shape[1] else 0
        w_packed[:, i//2] = (val0 & 0x0F) | ((val1 & 0x0F) << 4)
    w_packed = w_packed.reshape(shape[0], shape[1] // 2)
    scales = scales.reshape(shape[0], shape[1] // group_size)
    zeros = zeros.reshape(shape[0], shape[1] // group_size)
    return w_packed, scales, zeros

def unpack_int4(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    shape = weights.shape
    w_packed = weights.reshape(-1, shape[-1])
    scales = scales.reshape(-1, scales.shape[-1])
    zeros = zeros.reshape(-1, zeros.shape[-1])
    w_unpacked = torch.zeros(w_packed.shape[0], w_packed.shape[1] * 2, dtype=torch.float32, device=weights.device)
    for i in range(w_packed.shape[0]):
        group_idx = i // (shape[-1] * 2 // group_size)
        for j in range(w_packed.shape[1]):
            val = w_packed[i, j]
            low = val & 0x0F
            high = (val >> 4) & 0x0F
            w_unpacked[i, j*2] = low * scales[i, group_idx] + zeros[i, group_idx]
            w_unpacked[i, j*2+1] = high * scales[i, group_idx] + zeros[i, group_idx]
    return w_unpacked.reshape(shape[0], shape[1] * 2)


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
