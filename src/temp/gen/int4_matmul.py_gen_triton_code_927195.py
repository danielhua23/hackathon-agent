
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    bs_ptr, bzp_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_offset = k * BLOCK_SIZE_K * SPLIT_K
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] + k_offset < K)
        mask_b = (offs_k[:, None] + k_offset < K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs + k_offset * stride_ak, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs + (k_offset // 8) * stride_bk, mask=mask_b, other=0.0)

        group_idx = (offs_k[:, None] + k_offset) // group_size
        bs_ptrs = bs_ptr + group_idx * stride_bsk + offs_bn[None, :] * stride_bsn
        bzp_ptrs = bzp_ptr + group_idx * stride_bzpk + (offs_bn[None, :] // 8) * stride_bzpn

        bs = tl.load(bs_ptrs, mask=mask_b, other=0.0)
        bzp = tl.load(bzp_ptrs, mask=mask_b, other=0.0)

        b_shift = ((offs_k[:, None] + k_offset) % 8) * 4
        bzp_shift = (offs_bn[None, :] % 8) * 4

        int4_b = (b >> b_shift) & 0xF
        int4_bzp = (bzp >> bzp_shift) & 0xF

        fp_b = ((int4_b - int4_bzp) * bs).to(tl.float16)
        accumulator += tl.dot(a.to(tl.float16), fp_b)

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, c, mask=mask_c)
    else:
        tl.store(c_ptrs, c, mask=mask_c)

def matmul_dequantize_int4_s2(
    x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous(), "Input x must be contiguous"
    assert qweight.is_contiguous(), "qweight must be contiguous"
    assert scales.is_contiguous(), "scales must be contiguous"
    assert zeros.is_contiguous(), "zeros must be contiguous"

    M, K = x.shape
    N = scales.shape[1]

    output = torch.empty((M, N), device=x.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )
    matmul_kernel[grid](
        x, qweight, output,
        scales, zeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        group_size,
        GROUP_SIZE_M=8,
        SPLIT_K=1,
    )
    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=4),
    ],
    key=['num_rows', 'num_cols'],
)
@triton.jit
def quantize_int4_kernel(
    src_ptr, dst_ptr, scales_ptr, zeros_ptr,
    num_rows, num_cols,
    stride_sr, stride_sc,
    stride_dr, stride_dc,
    stride_scale,
    BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    cols_per_int32 = 8

    group_start = group * GROUP_SIZE
    group_end = tl.minimum(group_start + GROUP_SIZE, num_cols)
    num_ints = (GROUP_SIZE + cols_per_int32 - 1) // cols_per_int32

    col_offsets = group_start + tl.arange(0, BLOCK_SIZE)

    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)

    for offset in range(0, GROUP_SIZE, BLOCK_SIZE):
        mask = (col_offsets + offset) < group_end
        src_offs = src_ptr + row * stride_sr + (col_offsets + offset) * stride_sc
        vals = tl.load(src_offs, mask=mask, other=0.0)
        max_val = tl.where(mask, tl.maximum(max_val, vals), max_val)
        min_val = tl.where(mask, tl.minimum(min_val, vals), min_val)

    max_val = tl.max(max_val)
    min_val = tl.min(min_val)

    scale = (max_val - min_val) / 15.0
    zero = -min_val / scale

    scale_idx = row * (num_cols // GROUP_SIZE) + group
    tl.store(scales_ptr + scale_idx, scale.to(tl.float16))
    tl.store(zeros_ptr + scale_idx, zero.to(tl.float16))

    for offset in range(0, GROUP_SIZE, BLOCK_SIZE):
        mask = (col_offsets + offset) < group_end
        src_offs = src_ptr + row * stride_sr + (col_offsets + offset) * stride_sc
        vals = tl.load(src_offs, mask=mask, other=0.0)

        q = tl.clamp((vals / scale + zero).to(tl.int32), 0, 15)

        int32_ptrs = dst_ptr + row * stride_dr + ((group_start + offset) // cols_per_int32) * stride_dc

        for i_offset in range(0, BLOCK_SIZE, cols_per_int32):
            i = offset + i_offset
            if i < GROUP_SIZE:
                packed = tl.full([1], 0, dtype=tl.int32)
                for ch in range(cols_per_int32):
                    idx = i_offset + ch
                    val = q[idx] if (group_start + i + ch) < num_cols else tl.full([], 0, dtype=tl.int32)
                    packed = tl.bitwise_or(packed, tl.left_shift(val & 0xF, ch * 4))
                addr = int32_ptrs + (i // cols_per_int32) * stride_dc
                tl.store(addr, packed)

def quantize_int4(weight: torch.Tensor, group_size: int = 128) -> tuple:
    assert weight.dim() == 2, "weight must be 2D"
    num_rows, num_cols = weight.shape
    group_size = min(group_size, num_cols)
    assert num_cols % group_size == 0

    packed = torch.empty(
        (num_rows, num_cols // 8),
        dtype=torch.int32,
        device=weight.device
    )
    scales = torch.empty(
        (num_rows, num_cols // group_size),
        dtype=torch.float16,
        device=weight.device
    )
    zeros = torch.empty_like(scales)

    def grid():
        return (
            num_rows,
            num_cols // group_size,
        )

    quantize_int4_kernel[grid](
        weight, packed, scales, zeros,
        num_rows, num_cols,
        weight.stride(0), weight.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0),
        GROUP_SIZE=group_size,
    )
    return packed, scales, zeros

def unpack_int4(qw_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    assert qw_packed.dim() == 2 and scales.dim() == 2 and zeros.dim() == 2
    num_rows = qw_packed.size(0)
    num_cols = scales.size(1) * group_size
    weight = torch.empty((num_rows, num_cols), dtype=torch.float16, device=qw_packed.device)

    for row in range(num_rows):
        for group in range(scales.size(1)):
            scale = scales[row, group].item()
            zero = zeros[row, group].item()
            start_col = group * group_size
            end_col = start_col + group_size
            for col in range(start_col, end_col, 8):
                if (col // 8) >= qw_packed.size(1):
                    continue
                packed = qw_packed[row, col // 8].item()
                for k in range(8):
                    val = (packed >> (4 * k)) & 0xF
                    rescaled = val * scale + zero
                    idx = col + k
                    if idx < num_cols:
                        weight[row, idx] = rescaled
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
