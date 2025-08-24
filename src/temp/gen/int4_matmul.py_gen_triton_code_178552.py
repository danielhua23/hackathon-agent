
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
    stride_scales_k, stride_scales_n,
    stride_zeros_k, stride_zeros_n,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(grid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn)

    scales_ptrs = scales_ptr + ((offs_k[None, :] // group_size) * stride_scales_k + offs_n[:, None] * stride_scales_n)
    zeros_ptrs = zeros_ptr + ((offs_k[None, :] // group_size) * stride_zeros_k + (offs_n[:, None] // 8) * stride_zeros_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_start = k * BLOCK_SIZE_K * SPLIT_K + pid_z * BLOCK_SIZE_K
        idx_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = idx_k[None, :] < K
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)

        idx_k_packed = (idx_k[None, :] // 8)
        b = tl.load(b_ptr + idx_k_packed * stride_bk + offs_n[None, :] * stride_bn, mask=mask_k, other=0)

        idx_g = (idx_k[None, :] // group_size)
        bs = tl.load(scales_ptr + idx_g * stride_scales_k + offs_n[None, :] * stride_scales_n, mask=mask_k, other=1.0)
        zs = tl.load(zeros_ptr + idx_g * stride_zeros_k + (offs_n[None, :] // 8) * stride_zeros_n, mask=mask_k, other=0.0)

        shift = (idx_k[None, :] % 8) * 4
        int4_val = (b >> shift) & 0xF
        zp4_val_all = zs & 0x0F0F0F0F
        zp4_val = (zs >> ((offs_n[None, :] % 8) * 4)) & 0xF
        b_fp = (int4_val - zp4_val) * bs
        accumulator += tl.dot(a, b_fp.to(a.dtype))

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, accumulator, mask=mask_c)
    else:
        tl.store(c_ptrs, accumulator, mask=mask_c)

def matmul_dequantize_int4_s2(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, K: int) -> torch.Tensor:
    M, _ = x.shape
    _, N = qweight.shape

    c = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K']
    )
    matmul_kernel[grid](
        x, qweight, c,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        128
    )
    return c

def quantize_int4(weights: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    shape = weights.shape
    w = weights.to(torch.float32).reshape(-1, group_size)
    w_min = w.min(dim=1, keepdim=True)[0]
    w_max = w.max(dim=1, keepdim=True)[0]
    scales = (w_max - w_min) / 15.0
    zeros = -w_min / scales
    w = torch.round((w - w_min) / scales).clamp(0, 15).to(torch.uint8)
    packed = torch.zeros(w.shape[0], (w.shape[1] + 1) // 2, dtype=torch.int32, device=weights.device)
    packed[:, :w.shape[1]//2] = (w[:, ::2] & 0x0F) | ((w[:, 1::2] & 0x0F) << 4)
    packed = packed.reshape(shape[0], shape[1] // 8)
    scales = scales.reshape(shape[0], shape[1] // group_size)
    zeros = zeros.reshape(shape[0], shape[1] // group_size)
    return packed, scales, zeros, None

def unpack_int4(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    shape = weights.shape
    assert shape[-1] == scales.shape[-1]
    w = weights.view(-1, weights.shape[-1])
    scales = scales.view(-1, scales.shape[-1])
    zeros = zeros.view(-1, zeros.shape[-1])
    unpacked = torch.zeros(w.shape[0], w.shape[1] * 8, dtype=torch.float32, device=weights.device)
    for i in range(8):
        unpacked[:, i::8] = ((w >> (4*i)) & 0x0F) * scales[:, w.shape[1]*(8*i)//group_size:w.shape[1]*(8*i)//group_size+1].expand(-1, unpacked.shape[1]//8) + \
                            zeros[:, w.shape[1]*(8*i)//group_size:w.shape[1]*(8*i)//group_size+1].expand(-1, unpacked.shape[1]//8)
    return unpacked.reshape(weights.shape[0], weights.shape[1] * 8)


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
