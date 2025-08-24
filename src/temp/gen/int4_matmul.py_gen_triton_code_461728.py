
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_offset = k * BLOCK_SIZE_K * SPLIT_K
        a_idx = offs_k[None, :] + k_offset
        b_idx = offs_k[:, None] + k_offset
        mask_a = (offs_m[:, None] < M) & (a_idx < K)
        mask_b = (b_idx < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs + k_offset * stride_ak, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs + k_offset * stride_bk, mask=mask_b, other=0.0)

        accumulator += tl.dot(a, b)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, accumulator, mask=mask_c)
    else:
        tl.store(c_ptrs, accumulator.to(tl.float16), mask=mask_c)

def matmul_dequantize_int4_s2(
    x: torch.Tensor, qw_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor
) -> torch.Tensor:
    M, K = x.shape
    N = qw_packed.shape[0] * 8 // 4
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            1,
        )

    matmul_kernel[grid](
        x, qw_packed, y,
        M, N, K,
        x.stride(0), x.stride(1),
        qw_packed.stride(0), 4,
        y.stride(0), y.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        SPLIT_K=1,
    )
    return y

@triton.jit
def quantize_int4_kernel(
    src_ptr, dst_ptr, scales_ptr, zeros_ptr,
    num_rows, num_cols,
    stride_sr, stride_sc,
    stride_dr, stride_dc,
    stride_scale_r,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_start = tl.program_id(1) * GROUP_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)

    mask = offs < num_cols
    src_ptrs = src_ptr + row * stride_sr + offs * stride_sc
    src = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)

    min_val = tl.min(src)
    max_val = tl.max(src)
    scale = (max_val - min_val) / ((2 ** 4) - 1)
    zero = -min_val / scale
    scale_store = scale.to(tl.float16)
    zero_store = zero.to(tl.float16)

    grouped = (src - min_val) / scale
    int4 = tl.cast(grouped + 0.5, tl.int32)
    packed = (int4 & 0xF) | (tl.shl(int4, 4) & 0xF)
    packed = tl.view(packed, tl.int32)

    scale_zero_idx = row + (col_start // GROUP_SIZE) * stride_scale_r
    scales_ptrs = scales_ptr + scale_zero_idx
    zeros_ptrs = zeros_ptr + scale_zero_idx

    tl.store(scales_ptrs, scale_store)
    tl.store(zeros_ptrs, zero_store)

    if col_start < num_cols:
        src_ptrs = src_ptr + row * stride_sr + col_start * stride_sc
        for j in range(0, tl.cdiv(GROUP_SIZE, BLOCK_SIZE)):
            offset = j * BLOCK_SIZE
            mask = (col_start + offset + tl.arange(0, BLOCK_SIZE)) < num_cols
            src = tl.load(src_ptrs + offset * stride_sc, mask=mask, other=0.0).to(tl.float32)
            rescaled = (src - min_val) / scale
            int4 = tl.cast(rescaled + 0.5, tl.int32)
            packed = tl.zeros([BLOCK_SIZE // 8], dtype=tl.int32)
            for k in range(0, BLOCK_SIZE // 8):
                idx = k * 8 + tl.arange(0, 8)
                packed[k] = (
                    (int4[idx] & 0xF) |
                    tl.shl((int4[idx + 1] & 0xF), 4) |
                    tl.shl((int4[idx + 2] & 0xF), 8) |
                    tl.shl((int4[idx + 3] & 0xF), 12) |
                    tl.shl((int4[idx + 4] & 0xF), 16) |
                    tl.shl((int4[idx + 5] & 0xF), 20) |
                    tl.shl((int4[idx + 6] & 0xF), 24) |
                    tl.shl((int4[idx + 7] & 0xF), 28)
                )
            dst_ptrs = dst_ptr + row * stride_dr + (offset // 8) * stride_dc
            write_mask = (col_start + offset) < num_cols
            tl.store(dst_ptrs, packed, mask=write_mask)

def quantize_int4(weight: torch.Tensor, group_size: int = 128) -> tuple:
    assert weight.dim() == 2
    num_rows, num_cols = weight.shape
    group_size = min(group_size, num_cols)
    assert num_cols % group_size == 0
    num_groups = num_cols // group_size

    qw_packed = torch.empty(
        (num_rows, num_cols // 8),
        dtype=torch.int32,
        device=weight.device
    )
    scales = torch.empty(
        (num_rows, num_groups),
        dtype=torch.float16,
        device=weight.device
    )
    zeros = torch.empty_like(scales)

    def grid():
        return (
            num_rows,
            num_groups,
        )

    quantize_int4_kernel[grid](
        weight, qw_packed, scales, zeros,
        num_rows, num_cols,
        weight.stride(0), weight.stride(1),
        qw_packed.stride(0), qw_packed.stride(1),
        scales.stride(0),
        BLOCK_SIZE=32,
        GROUP_SIZE=group_size,
    )
    return qw_packed, scales, zeros

def unpack_int4(qw_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    num_rows = qw_packed.size(0)
    num_cols = qw_packed.size(1) * 8
    weight = torch.empty((num_rows, num_cols), dtype=torch.float16, device=qw_packed.device)
    for row in range(num_rows):
        group_idx = 0
        for col in range(0, num_cols, 8):
            packed = qw_packed[row, col // 8]
            scale = scales[row, group_idx]
            zero = zeros[row, group_idx]
            if (col + 8) % group_size == 0:
                group_idx += 1
            for i in range(8):
                val = (packed >> (4 * i)) & 0xF
                rescaled = val * scale + zero
                weight[row, col + i] = rescaled
    return weight


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
