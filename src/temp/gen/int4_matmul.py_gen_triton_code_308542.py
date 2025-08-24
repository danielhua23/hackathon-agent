
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
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, M - first_pid_m * BLOCK_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= M or pid_n * BLOCK_SIZE_N >= N:
        return

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a_mask = offs_am[:, None] < M and offs_k[None, :] < K
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        offs_k_in_group = offs_k // GROUP_SIZE
        scales = tl.load(scales_ptr + offs_bn[None, :] * stride_scales_n + offs_k_in_group[:, None] * stride_scales_g)
        zeros = tl.load(zeros_ptr + offs_bn[None, :] * stride_zeros_n + offs_k_in_group[:, None] * stride_zeros_g)

        b = tl.load(b_ptrs, mask=offs_k[:, None] < K and offs_bn[None, :] < N, other=0.0)
        b = b.to(tl.int32)

        b0 = (b & 0x0F) - 8
        b1 = ((b >> 4) & 0x0F) - 8

        dequant_b0 = b0.to(tl.float32) * scales + zeros
        dequant_b1 = b1.to(tl.float32) * scales + zeros

        b_reconstructed = tl.zeros((BLOCK_SIZE_K * 2, BLOCK_SIZE_N), dtype=tl.float32)
        b_reconstructed = tl.where(tl.arange(0, BLOCK_SIZE_K * 2)[:, None] % 2 == 0,
                                   dequant_b0[tl.arange(0, BLOCK_SIZE_K)[:, None], :],
                                   dequant_b1[tl.arange(0, BLOCK_SIZE_K)[:, None], :])

        valid_k = min(BLOCK_SIZE_K * 2, K - k * BLOCK_SIZE_K * 2)
        a_inner = a[:, :valid_k]
        b_inner = b_reconstructed[:valid_k, :]

        acc += tl.dot(a_inner, b_inner)

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
        offs_k += BLOCK_SIZE_K * SPLIT_K

    if SPLIT_K > 1:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        mask = offs_cm[:, None] < M and offs_cn[None, :] < N
        tl.atomic_add(c_ptrs, acc, mask=mask)
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        mask = offs_cm[:, None] < M and offs_cn[None, :] < N
        tl.store(c_ptrs, acc, mask=mask)


def quantize_int4(x: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    rows, cols = x.shape
    assert cols % group_size == 0
    num_groups = cols // group_size

    x_groups = x.view(rows, num_groups, group_size)
    x_min = x_groups.min(dim=2, keepdim=True)[0]
    x_max = x_groups.max(dim=2, keepdim=True)[0]
    scale = (x_max - x_min) / 15.0
    zero = -x_min / scale

    x_quantized = ((x_groups - x_min) / scale).round().clamp(0, 15).to(torch.int32) - 8
    x_quantized_uint = (x_quantized + 8).to(torch.uint8)

    packed = torch.zeros(rows, num_groups, group_size // 2, dtype=torch.int32, device=x.device)
    for j in range(group_size // 2):
        idx = j * 2
        packed[:, :, j] = (
            (x_quantized_uint[:, :, idx] & 0x0F) |
            ((x_quantized_uint[:, :, idx + 1] & 0x0F) << 4)
        )

    scales = scale.squeeze(-1).contiguous()
    zeros = zero.squeeze(-1).contiguous()
    return packed.view(rows, num_groups * group_size // 2), scales, zeros


def matmul_dequantize_int4_s2(
    a: torch.Tensor, b_q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
    group_size: int = 128, split_k: int = 1
) -> torch.Tensor:
    assert a.dim() == 2
    assert b_q.dim() == 2
    assert scales.dim() == 2
    assert zeros.dim() == 2
    assert a.shape[1] == b_q.shape[0] * 2, "Dimension mismatch between A and quantized B"
    M, K = a.shape
    N, _ = scales.shape

    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    if split_k > 1:
        c.zero_()

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        split_k,
    )

    matmul_kernel[grid](
        a, b_q, c,
        scales, zeros,
        M, N, K,
        a.stride(0), a.stride(1),
        b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        SPLIT_K=split_k,
    )
    return c


def unpack_int4(b_q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    assert b_q.dim() == 2
    assert scales.dim() == 2
    assert zeros.dim() == 2
    rows, cols_packed = b_q.shape
    assert cols_packed * 2 == scales.shape[1] * group_size

    cols = cols_packed * 2
    b_unpacked = torch.zeros(rows, cols, dtype=torch.float32, device=b_q.device)

    scales_expanded = scales.repeat_interleave(group_size, dim=1)
    zeros_expanded = zeros.repeat_interleave(group_size, dim=1)

    for j in range(cols_packed):
        packed_col = b_q[:, j]
        idx = j * 2
        b_unpacked[:, idx] = ((packed_col & 0x0F).to(torch.float32) - 8) * scales_expanded[:, idx] + zeros_expanded[:, idx]
        if idx + 1 < cols:
            b_unpacked[:, idx + 1] = (((packed_col >> 4) & 0x0F).to(torch.float32) - 8) * scales_expanded[:, idx + 1] + zeros_expanded[:, idx + 1]

    return b_unpacked


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
