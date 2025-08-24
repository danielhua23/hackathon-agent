
import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256,'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128,'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128,'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=2,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2}, num_stages=3,num_warps=8),
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
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_z * (BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk) + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_off = k * BLOCK_SIZE_K * SPLIT_K
        k_now = k_off + offs_k

        a_trans_mask = k_now[None, :] < K
        a = tl.load(a_ptrs, mask=a_trans_mask, other=0.0)

        b_pack_idx = (k_now[:, None] // 8)
        b_n_idx = offs_n = offs_bn[None, :]
        b_load_mask = k_now[:, None] < K
        b_pack = tl.load(b_ptrs, mask=b_load_mask, other=0)

        g_idx = (k_now[:, None] // group_size)
        bs = tl.load(
            bs_ptr + g_idx * stride_bsk + b_n_idx * stride_bsn,
            mask=b_load_mask, other=0.0
        )

        zp_idx = (b_n_idx // 8)
        bzp_pack = tl.load(
            bzp_ptr + g_idx * stride_bzpk + zp_idx * stride_bzpn,
            mask=b_load_mask, other=0
        )

        b_shift = (k_now[:, None] % 8) * 4
        bzp_shift = (b_n_idx % 8) * 4
        b_int4 = (b_pack >> b_shift) & 0xF
        bzp_int4 = (bzp_pack >> bzp_shift) & 0xF
        b_float = (b_int4 - bzp_int4) * bs
        accumulator += tl.dot(a, b_float.to(a.dtype))

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk

    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=mask)


def matmul_dequantize_int4_s2(
    x: torch.FloatTensor,
    qweight: torch.IntTensor,
    scales: torch.FloatTensor,
    qzeros: torch.IntTensor,
    group_size: int = 128,
    output=None
) -> torch.FloatTensor:
    assert x.is_contiguous(), "x must be contiguous"
    assert qweight.is_contiguous(), "qweight must be contiguous"

    M, K = x.shape
    N = scales.shape[1]

    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )
    matmul_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size,
    )
    return output


def quantize_int4(
    w: torch.Tensor,
    group_size: int = 128
):
    w = w.float()
    K, N = w.shape
    assert K % group_size == 0, "K must be divisible by group_size"
    w = w.view(K // group_size, group_size, N)

    wmin = w.min(dim=1, keepdim=True)[0]
    wmax = w.max(dim=1, keepdim=True)[0]
    scale = (wmax - wmin) / 15.0
    zero = torch.round(-wmin / scale).clamp(0, 15).to(torch.uint8)

    int4 = torch.round((w - wmin) / scale).clamp(0, 15).to(torch.uint8)

    int4 = int4.view(K, N)
    zero = zero.view(K // group_size, N)

    packed = torch.zeros((K, N // 8), dtype=torch.int32, device=w.device)
    for col in range(0, N, 8):
        val = (
            int4[:, col + 7] << 28 |
            int4[:, col + 6] << 24 |
            int4[:, col + 5] << 20 |
            int4[:, col + 4] << 16 |
            int4[:, col + 3] << 12 |
            int4[:, col + 2] << 8  |
            int4[:, col + 1] << 4  |
            int4[:, col + 0]
        ).to(torch.int32)
        packed[:, col // 8] = val

    zero_packed = torch.zeros((K // group_size, N // 8), dtype=torch.int32, device=w.device)
    for col in range(0, N, 8):
        zval = (
            zero[:, col + 7] << 28 |
            zero[:, col + 6] << 24 |
            zero[:, col + 5] << 20 |
            zero[:, col + 4] << 16 |
            zero[:, col + 3] << 12 |
            zero[:, col + 2] << 8  |
            zero[:, col + 1] << 4  |
            zero[:, col + 0]
        ).to(torch.int32)
        zero_packed[:, col // 8] = zval

    return packed, scale.half(), zero_packed


def unpack_int4(
    packed: torch.IntTensor,
    scale: torch.FloatTensor,
    zero: torch.IntTensor,
    group_size: int = 128
):
    K, Nw = packed.shape
    N = Nw * 8
    Kg = K // group_size
    scale = scale.view(Kg, 1, N).expand(-1, group_size, -1).reshape(K, N)
    zero_exp = zero.view(Kg, 1, Nw).expand(-1, group_size, -1).reshape(K, Nw)
    zero_bytes = torch.empty_like(packed, dtype=torch.uint8).repeat_interleave(2, dim=1)[:, :N]
    for col in range(N):
        idx = col // 8
        shift = (col % 8) * 4
        zero_bytes[:, col] = ((zero_exp[:, idx] >> shift) & 0xF)
    zero = zero_bytes
    unpacked = torch.empty_like(packed, dtype=torch.uint8).repeat_interleave(8, dim=1)[:, :N]
    for col in range(N):
        idx = col // 8
        shift = (col % 8) * 4
        unpacked[:, col] = ((packed[:, idx] >> shift) & 0xF)
    return (unpacked.float() - zero.float()) * scale.float()


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
