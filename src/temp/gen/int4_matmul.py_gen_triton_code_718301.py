
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4)
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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
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
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        bs_ptrs = bs_ptr + ((offs_k[None, :] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bsk \
            + offs_n[None, :] * stride_bsn
        bzp_ptrs = bzp_ptr + ((offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size) * stride_bzpk \
            + (offs_n[None, :] // 8) * stride_bzpn
        b_shift_bits = (offs_k[:, None] % 8) * 4
        z_shift_bits = (offs_n[None, :] % 8) * 4
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
        bs = tl.load(bs_ptrs, mask=offs_n[None, :] < N, other=0.0)
        bzp = tl.load(bzp_ptrs, mask=offs_n[None, :] < N, other=0)
        b_q = ((b >> b_shift_bits) & 0xF)
        z_q = ((bzp >> z_shift_bits) & 0xF)
        b_deq = ((b_q.to(tl.float32) - z_q.to(tl.float32)) * bs).to(a.dtype)
        accumulator += tl.dot(a, b_deq)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptr + offs_cm, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptr + offs_cm, c, mask=c_mask)

def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int = 128, output: torch.FloatTensor = None) -> torch.FloatTensor:
    assert x.is_contiguous(), "input must be contiguous"
    M, K = x.shape
    N = scales.shape[1]
    assert K == qweight.shape[0] * 8, "Input K must match qweight shape"
    assert N == qweight.shape[1], "Input N must match qweight shape"
    assert scales.shape[0] == (K + group_size - 1) // group_size, "Scales shape mismatch"
    assert qzeros.shape[0] == (K + group_size - 1) // group_size, "Qzeros shape mismatch"
    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K']
    )
    matmul_kernel[grid](
        x, qweight, output, scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size
    )
    return output

def quantize_int4(x: torch.Tensor, group_size: int = 128):
    x = x.t().contiguous()
    N, K = x.shape
    assert K % group_size == 0

    x = x.view(N, K // group_size, group_size).float()
    x_min = x.min(dim=2, keepdim=True)[0]
    x_max = x.max(dim=2, keepdim=True)[0]

    scales = (x_max - x_min) / 15.0
    zeros = (-x_min / scales).round().clamp(0, 15)
    x_q = (x / scales + zeros).round().clamp(0, 15)
    scales = scales.squeeze(2).t().contiguous()
    zeros = zeros.squeeze(2).long().t().contiguous()

    x_q = x_q.view(N, K)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=x.device)
    for i in range(8):
        packed |= ((x_q[:, i::8]).to(torch.int32) & 0xF) << (4 * i)
    packed = packed.t().contiguous()

    zeros = zeros.view(scales.shape)
    return packed, scales.float(), zeros

def unpack_int4(qweight: torch.IntTensor, scales: torch.FloatTensor, zeros: torch.FloatTensor, group_size: int = 128) -> torch.FloatTensor:
    qweight, scales, zeros = qweight.t(), scales.t(), zeros.t()
    N, K_w = qweight.shape
    K = K_w * 8
    weight = torch.zeros((N, K), dtype=torch.float32, device=qweight.device)

    for i in range(8):
        mask = 0xF << (i * 4)
        cols = torch.arange(i, K, 8, device=qweight.device)
        g_idx = cols // group_size
        scale = scales[:, g_idx]
        zero = zeros[:, g_idx]
        vals = ((qweight & mask) >> (i * 4)).to(torch.float32)
        weight[:, cols] = (vals - zero) * scale

    return weight.t()


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
