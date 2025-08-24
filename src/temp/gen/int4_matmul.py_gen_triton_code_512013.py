
import torch
import triton
import triton.language as tl

# -------------------------------------------------------------------------
# Triton kernel – core INT4 matmul
# -------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=2),
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
        k_offs = k * BLOCK_SIZE_K * SPLIT_K + offs_k
        ks = bs_ptr + (k_offs // group_size) * stride_bsk + offs_n[None, :] * stride_bsn
        kzp = bzp_ptr + (k_offs // group_size) * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn
        a = tl.load(a_ptrs, mask=k_offs[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=k_offs[:, None] < K, other=0)
        scale = tl.load(ks)
        zero = tl.load(kzp)
        b_shift = (k_offs[:, None] % 8) * 4
        z_shift = (offs_n[None, :] % 8) * 4
        b_deq = (((b >> b_shift) & 0xF).to(tl.float32) - ((zero >> z_shift) & 0xF).to(tl.float32)) * scale
        accumulator += tl.dot(a.to(tl.float16), b_deq.to(tl.float16))
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K // 8 * stride_bk
    c = accumulator.to(tl.float16)

    if SPLIT_K > 1:
        offs_cm = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.atomic_add(c_ptr + offs_cm, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    else:
        offs_cm = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptr + offs_cm, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# -------------------------------------------------------------------------
# Python wrapper
# -------------------------------------------------------------------------
def matmul_dequantize_int4_s2(
    x: torch.FloatTensor,
    qweight: torch.IntTensor,
    scales: torch.FloatTensor,
    qzeros: torch.IntTensor,
    group_size: int = 128,
    output: torch.FloatTensor = None
) -> torch.FloatTensor:
    assert x.is_contiguous(), "input must be contiguous"
    M, K = x.shape
    N = scales.shape[1]
    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K']
    )
    matmul_kernel[grid](
        x, qweight, output, scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(1), qweight.stride(0),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size
    )
    return output

# -------------------------------------------------------------------------
# Quantization / De-quantization helpers
# -------------------------------------------------------------------------
def quantize_int4(x: torch.Tensor, group_size: int = 128):
    """
    Converts fp16/fp32 weight tensor of shape (K, N) into INT4 representation.
    Returns (packed_int32, scales, zeros) all on same device/dtype
    layout expected by the kernel.
    """
    x = x.t().contiguous()          # -> (N, K)
    N, K = x.shape
    assert K % group_size == 0, f"K ({K}) not divisible by group_size {group_size}"

    x = x.view(N, K // group_size, group_size).float()
    x_min = x.min(dim=2, keepdim=True)[0]
    x_max = x.max(dim=2, keepdim=True)[0]

    scales = (x_max - x_min) / 15.0
    zp_fp = (-x_min / scales).round().clamp(0, 15)
    x_q = (x / scales + zp_fp).round().clamp(0, 15)
    scales = scales.squeeze(2).t().contiguous()
    zeros = zp_fp.squeeze(2).t().contiguous()

    # Flatten to (N, K) before packing
    x_q = x_q.view(N, K)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=x.device)
    for i in range(8):
        packed |= ((x_q[:, i::8].to(torch.int32) & 0xF) << (4 * i))
    packed = packed.t().contiguous()
    return packed, scales, zeros


def unpack_int4(qweight: torch.IntTensor,
                scales: torch.FloatTensor,
                zeros: torch.FloatTensor,
                group_size: int = 128) -> torch.FloatTensor:
    """
    Unpack INT4 tensor back into float32/float16 tensor for correctness test.
    """
    qweight, scales, zeros = qweight.t(), scales.t(), zeros.t()
    N, K_w = qweight.shape
    K = K_w * 8
    weight = torch.zeros((N, K), dtype=torch.float32, device=qweight.device)

    for i in range(8):
        mask = 0xF << (i * 4)
        cols = torch.arange(i, K, 8, device=qweight.device)
        scale_col = scales[:, cols // group_size]
        zero_col = zeros[:, cols // group_size]
        vals = ((qweight & mask) >> (i * 4)).to(torch.float32)
        weight[:, cols] = (vals - zero_col) * scale_col
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
