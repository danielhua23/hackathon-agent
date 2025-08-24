
import torch
import triton
import triton.language as tl

# -----------------  Triton Kernel (INT4 matrix multiply) -----------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1},
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                       'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M':  32, 'BLOCK_SIZE_N': 32,
                       'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'SPLIT_K': 1},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M':  32, 'BLOCK_SIZE_N': 32,
                       'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16, 'SPLIT_K': 2},
                      num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
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
    offs_k_start = pid_k * BLOCK_SIZE_K
    offs_k = offs_k_start + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn

    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        cur_k = offs_k_start + k * BLOCK_SIZE_K * SPLIT_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = cur_k[None, :] < K
        mask_n = offs_n[None, :] < N
        load_a = tl.load(a_ptrs, mask=mask_k & (offs_m[:, None] < M), other=0.0)
        packed_b = tl.load(b_ptrs, mask=mask_k & mask_n, other=0)
        packed_b = packed_b.to(tl.int32)

        group_idx = cur_k[None, :] // group_size
        scale_ptr = scales_ptr + offs_n[None, :] * stride_bsn
        zero_ptr  = zeros_ptr  + (offs_n[None, :] // 8) * stride_bzpn
        scale_ptr += group_idx * stride_bsk
        zero_ptr  += group_idx * stride_bzpk

        scale = tl.load(scale_ptr, mask=mask_k & mask_n, other=0.0)
        zero_packed = tl.load(zero_ptr, mask=mask_k & mask_n, other=0)
        zero_packed = zero_packed.to(tl.int32)

        shift = (cur_k[None, :] % 8) * 4
        zp_shift = (offs_n[None, :] % 8) * 4

        int_b = (packed_b >> shift) & 0xF
        int_zp = (zero_packed >> zp_shift) & 0xF
        b = ((int_b.to(tl.float32) - int_zp.to(tl.float32)) * scale)
        accum += tl.dot(load_a, b)

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K * stride_bk // 8)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, accum, mask=mask)
    else:
        tl.atomic_add(c_ptrs, accum, mask=mask)

# ----------------- Python helpers ----------------------------------------

def matmul_dequantize_int4_s2(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    assert x.is_contiguous()
    assert qweight.is_contiguous()

    M, K = x.shape
    Kw, N = qweight.shape
    assert Kw == K // 2, "Packed weight shape mismatch"
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

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
    )
    return output

def quantize_int4(weights: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weights.dim() == 2
    K, N = weights.shape
    assert K % group_size == 0
    num_groups = K // group_size

    flat = weights.float().view(num_groups, group_size, N)
    mn, mx = flat.aminmax(dim=1)
    scale = (mx - mn) / 15.0
    scale = torch.where(scale == 0, 1.0, scale)
    zero = (-mn / scale)
    q = ((flat / scale[:, :, None] + zero[:, :, None] + 0.5).floor()).clamp(0, 15).to(torch.int32)

    q = q.view(K, N)
    q_low = q[::2]
    q_high = q[1::2]
    packed = (q_low & 0xF) | ((q_high & 0xF) << 4)

    return packed, scale.squeeze(1), zero.squeeze(1)

def unpack_int4(w: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    assert w.dtype == torch.int32
    K_qua, N = w.shape
    K = K_qua * 2
    assert K % group_size == 0
    num_groups = K // group_size

    w0 = (w & 0xF).to(torch.float32)
    w1 = ((w >> 4) & 0xF).to(torch.float32)
    unpacked = torch.zeros(K, N, dtype=torch.float32, device=w.device)
    unpacked[::2] = w0
    unpacked[1::2] = w1

    scales = scales.view(num_groups, 1, N).repeat(1, group_size, 1).reshape(K, N)
    zeros = zeros.view(num_groups, 1, N).repeat(1, group_size, 1).reshape(K, N)
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
