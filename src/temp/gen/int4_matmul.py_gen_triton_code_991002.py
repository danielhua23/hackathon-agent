
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
    offs_m = tl.where(offs_m < M, offs_m, 0)
    offs_n = tl.where(offs_n < N, offs_n, 0)

    offs_k_step = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_step[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k_step[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_idx in range(0, num_pid_k):
        kk = k_idx * BLOCK_SIZE_K * SPLIT_K
        mask_k = kk + offs_k_step[None, :]

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (mask_k < K), other=0.0)
        b = tl.load(b_ptrs, mask=mask_k < K, other=0)

        offs_gp = (kk + offs_k_step)[None, :] // group_size
        bs_ptrs = bs_ptr + offs_gp * stride_bsk + offs_n[None, :] * stride_bsn
        bzp_ptrs = bzp_ptr + offs_gp * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn

        bs = tl.load(bs_ptrs, mask=mask_k < K, other=1.0)
        bzp = tl.load(bzp_ptrs, mask=mask_k < K, other=0)

        mask_n = offs_n[None, :]
        shift = (kk + offs_k_step)[:, None] % 8 * 4
        shift_zp = mask_n % 8 * 4

        nib = (b >> shift) & 0xF
        zp = (bzp >> shift_zp) & 0xF
        fp_b = (nib - zp) * bs
        acc += tl.dot(a, fp_b.to(a.dtype))

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_cm < M
    mask_n = offs_cn < N
    offs_cm = tl.where(mask_m, offs_cm, 0)
    offs_cn = tl.where(mask_n, offs_cn, 0)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask)


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
    scales = scales.contiguous()
    qzeros = qzeros.contiguous()
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        1,
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
    w = weights.to(torch.float32)
    K, N = w.shape
    assert K % group_size == 0

    w_ = w.view(K // group_size, group_size, N)
    wmin = w_.amin(dim=1, keepdim=True)
    wmax = w_.amax(dim=1, keepdim=True)
    scales = ((wmax - wmin) / 15.0).squeeze(1)
    zeros = ((-wmin) / scales).round().clamp(0, 15).squeeze(1)

    quantized = torch.round((w_ - wmin) / scales).clamp(0, 15).to(torch.uint8)
    quantized = torch.bitwise_and(quantized, 0xF)
    quantized = quantized.view(K, N)

    packed = torch.zeros(K, N // 8, dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (quantized[:, i::8] << (4 * i)).to(torch.int32)
    return packed.contiguous(), scales.contiguous(), zeros.contiguous(), None


def unpack_int4(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    Kq, N = weights.shape
    K = Kq * 1
    unpacked = torch.zeros(K, N * 8, device=weights.device, dtype=scales.dtype)
    w_flat = weights.view(-1)
    for b in range(8):
        nib = (w_flat >> (b * 4)) & 0xF
        unpacked.view(-1)[b::8] = nib.float()
    unpacked = unpacked.view(K, N * 8)
    scales = scales.view(-1, N).repeat_interleave(group_size, dim=0)[:K]
    zeros = zeros.view(-1, N).repeat_interleave(group_size, dim=0)[:K]
    return (unpacked - zeros) * scales


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
