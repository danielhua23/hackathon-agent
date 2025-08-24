
import torch
import triton
import triton.language as tl

### -------------------  Triton kernel for INT4 matmul – autotuned  ------------------- ###
@triton.autotune(
    configs=[
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K:   tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    k_per_iter = BLOCK_SIZE_K * SPLIT_K

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k0 = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k0[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k0[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, k_per_iter)):
        idx_k = k * k_per_iter + pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = idx_k < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        b_offs = (idx_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
        b_int = tl.load(b_ptr + b_offs, mask=mask_k[:, None] & mask_n[None, :], other=0)

        g = idx_k[:, None] // group_size
        bs_offs = g * stride_bsk + offs_n[None, :] * stride_bsn
        bzp_offs = g * stride_bzpk + (offs_n[None, :]//8) * stride_bzpn
        bs = tl.load(bs_ptr + bs_offs, mask=g*0 == 0, other=1.0)
        bzp = tl.load(bzp_ptr + bzp_offs, mask=g*0 == 0, other=0)

        shift_k = (idx_k[:, None] % 8) * 4
        shift_n = (offs_n[None, :] % 8) * 4
        b_val = ((b_int >> shift_k) & 0xF) - ((bzp >> shift_n) & 0xF)
        b_fp = (b_val * bs).to(a.dtype)

        acc += tl.dot(a, b_fp)
        a_ptrs += k_per_iter * stride_ak

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    mask_c = mask_m[:, None] & mask_n[None, :]
    c_ptrs = c_ptr + offs_cm * stride_cm + offs_cn * stride_cn
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask_c)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask_c)

### -------------------  Launcher  ------------------- ###
def matmul_dequantize_int4_s2(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous()
    M, _ = x.shape
    N = scales.shape[1]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K']
    )
    matmul_kernel[grid](
        x, qweight, out,
        scales, qzeros,
        M, N, x.shape[1],
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size
    )
    return out

### -------------------  quantize / unpack  ------------------- ###
def quantize_int4(weights: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    w = weights.to(torch.float32)
    K, N = w.shape
    assert N % group_size == 0, "N must be divisible by group_size"
    w = w.view(-1, group_size)
    wmin = w.min(dim=1, keepdim=True)[0]
    wmax = w.max(dim=1, keepdim=True)[0]
    scales = (wmax - wmin) / 15.0
    zeros = (-wmin / scales).round().clamp(0, 15)

    w = torch.round((w - wmin) / scales).clamp(0, 15).to(torch.uint8)
    packed = torch.zeros(w.shape[0], w.shape[1] // 2, dtype=torch.int32, device=w.device)
    packed |= (w[:, 0::2] & 0xF)
    packed |= (w[:, 1::2] & 0xF) << 4
    packed = packed.view(K, N // 8)
    scales = scales.view(K, N // group_size).contiguous()
    zeros  =  zeros.view(K, N // group_size).contiguous()
    return packed.contiguous(), scales, zeros, None

def unpack_int4(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    Bits = 4
    Kq, N = weights.shape
    K = Kq * Bits
    unpacked = torch.empty(K, N, dtype=scales.dtype, device=weights.device)
    for b in range(Bits):
        int4 = (weights >> b) & 0x1
        row_off = b
        unpacked[row_off::Bits, :] = int4 * scales.view(-1, N).expand_as(int4) + zeros.view(-1, N).expand_as(int4)
    return unpacked.view(weights.shape[0] * 8, N)


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
