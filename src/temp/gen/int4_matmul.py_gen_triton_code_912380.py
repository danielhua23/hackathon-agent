
import torch
import triton
import triton.language as tl


##############################################
# Triton kernel(s)
##############################################

@triton.autotune(
    configs=[
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr']
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
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_k_step = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_off = offs_m[:, None] * stride_am + offs_k_step[None, :] * stride_ak
    b_off = (offs_k_step[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn

    bs_off = (offs_k_step[:, None] // group_size) * stride_bsk + offs_n[None, :] * stride_bsn
    bzp_off = (offs_k_step[:, None] // group_size) * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_idx in range(0, num_pid_k):
        kk = k_idx * BLOCK_SIZE_K * SPLIT_K
        mask_k = (kk + offs_k_step[None, :]) < K

        a = tl.load(a_ptr + a_off, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptr + b_off, mask=mask_k, other=0)

        bs = tl.load(bs_ptr + bs_off, mask=mask_k, other=1.0)
        bzp = tl.load(bzp_ptr + bzp_off, mask=mask_k, other=0)

        shift = (kk + offs_k_step)[None, :] % 8 * 4
        shift_zp = offs_n[None, :] % 8 * 4

        nib = (b >> shift) & 0xF
        z = (bzp >> shift_zp) & 0xF

        b_deq = (nib - z) * bs
        acc += tl.dot(a, b_deq.to(a.dtype))

        a_off += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_off += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk
        bs_off += (BLOCK_SIZE_K * SPLIT_K // group_size) * stride_bsk
        bzp_off += (BLOCK_SIZE_K * SPLIT_K // group_size) * stride_bzpk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask_c)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask_c)


##############################################
# Python wrappers
##############################################

def matmul_dequantize_int4_s2(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous()
    assert qweight.is_contiguous()
    M, K = x.shape
    N = scales.shape[1]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )
    matmul_kernel[grid](
        x, qweight, out,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size,
    )
    return out


def quantize_int4(weights: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """
    quantize weight matrix (K, N) to INT4 packed (K, N//8) plus per_group scale/zero_point.
    Scale / zero_point when per_group have shape (K//group_size, N).
    """
    w = weights.to(torch.float32)
    K, N = w.shape
    assert N % group_size == 0, "N must be divisible by group_size"

    w_ = w.view(-1, group_size)
    wmin = w_.min(dim=1, keepdim=True)[0]
    wmax = w_.max(dim=1, keepdim=True)[0]
    scales = (wmax - wmin) / 15.0
    zeros = (-wmin / scales).round().clamp(0, 15)

    quantized = torch.round((w_ - wmin) / scales).clamp(0, 15).to(torch.uint8)

    dw = quantized.shape[1]
    packed = torch.zeros(
        quantized.shape[0],
        (dw + 1) // 2,                   # two nibbles per byte
        dtype=torch.int32,
        device=quantized.device
    )

    idx_even = torch.arange(0, dw, 2, device=quantized.device)
    idx_odd  = torch.arange(1, dw, 2, device=quantized.device)

    packed_even = quantized[:, idx_even] & 0xF
    packed_odd  = (quantized[:, idx_odd] & 0xF) << 4
    packed[:, idx_even // 2] = packed_even + packed_odd

    packed = packed.view(K, N // 8)
    scales = scales.view(K, N // group_size).contiguous()
    zeros  = zeros.view(K, N // group_size).contiguous()
    return packed.contiguous(), scales, zeros, None


def unpack_int4(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """
    revert packed (K, N//8) into fp tensor (K*8, N).
    """
    Kq, N = weights.shape
    K, _ = scales.shape
    N_orig = scales.shape[1] * group_size

    unpacked = torch.empty(K, N * 8, dtype=scales.dtype, device=weights.device)

    flat = weights.view(-1)

    for b in range(8):
        nib = (flat >> (b * 4)) & 0xF
        unpacked[:, b::8] = nib.view(K, N)

    unpacked = unpacked.view(K, N * 8)
    scales = scales.view(-1, N).repeat_interleave(group_size, dim=0)
    zeros  = zeros.view(-1, N).repeat_interleave(group_size, dim=0)
    return (unpacked.float() - zeros) * scales


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
