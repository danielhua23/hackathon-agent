
import torch
import triton
import triton.language as tl

# =========================
# Triton kernel (batched INT4 matrix multiply)
# =========================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128,
                       'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                       'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
                       'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
                       'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16, 'SPLIT_K': 2}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
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
                  SPLIT_K: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)       # only meaningful when SPLIT_K > 1

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid0 // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid0 % group_size_m)
    pid_n = (pid0 % num_pid_in_group) // group_size_m

    # block row/col indices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    k_step = BLOCK_SIZE_K * SPLIT_K
    k_lo = pid_k * BLOCK_SIZE_K
    offs_k_block = k_lo + tl.arange(0, BLOCK_SIZE_K)

    # pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k_block[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k_block[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, k_step)):
        cur_k = offs_k_block + k * k_step
        mask_k = cur_k[None, :] < K
        mask_n = offs_n[None, :] < N

        a = tl.load(a_ptrs, mask=mask_k & (offs_m[:, None] < M), other=0.0)

        packed_b = tl.load(b_ptrs, mask=mask_k & mask_n, other=0)

        # group indices
        gidx = cur_k[None, :] // group_size

        scales = tl.load(scales_ptr +
                         gidx * stride_bsk +
                         offs_n[None, :] * stride_bsn, mask=mask_k & mask_n, other=0.0)

        zeros_packed = tl.load(zeros_ptr +
                               gidx * stride_bzpk +
                               (offs_n[None, :] // 8) * stride_bzpn,
                               mask=mask_k & mask_n, other=0)
        zeros_packed = zeros_packed.to(tl.int32)

        shift = (cur_k[None, :] % 8) * 4
        zp_shift = (offs_n[None, :] % 8) * 4

        int_b = (packed_b >> shift) & 0xF
        int_zp = (zeros_packed >> zp_shift) & 0xF
        b = ((int_b.to(tl.float32) - int_zp.to(tl.float32)) * scales)
        acc += tl.dot(a, b)

        a_ptrs += k_step * stride_ak
        b_ptrs += (k_step // 8) * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask_c)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask_c)

# =========================
# Front-end helpers
# =========================

def quantize_int4(weights: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize weights to INT4, packing 8 INT4 values per int32 row.
    Returns (qweight, scales, zeros) where
      - qweight: [Kw, N] int32，Kw = ceil_div(K, 8)
      - scales:  [num_groups, N] float
      - zeros:   [num_groups, N] int32 after packing (8 zeros per int32)
    """
    assert weights.dim() == 2
    K, N = weights.shape
    assert K % group_size == 0

    num_groups = K // group_size
    w_groups = weights.view(num_groups, group_size, N)          # [G, Gsz, N]
    w_min, w_max = w_groups.aminmax(dim=1)                      # [G, N]
    scale = (w_max - w_min) / 15.0
    scale = torch.where(scale == 0, torch.tensor(1.0, device=scale.device), scale)
    zero = (-w_min / scale)
    q = ((w_groups / scale.unsqueeze(1) + zero.unsqueeze(1) + 0.5).floor()).clamp(0, 15).to(torch.int32)

    q = q.view(K, N)                                            # [K, N]
    q_low = q[0::2]
    q_high = q[1::2]
    # pack into int32: [Kw, N]
    packed = (q_low & 0xF) | ((q_high & 0xF) << 4)

    # pack zeros similarly
    zero_int = zero.round().int().clip(0, 15)
    zero_low  = zero_int[..., 0::2]
    zero_high = zero_int[..., 1::2]
    zeros_packed = (zero_low & 0xF) | ((zero_high & 0xF) << 4)

    return packed, scale, zeros_packed


def unpack_int4(w, scales, zeros, group_size: int = 128):
    """
    De-quantize w for numeric validation.
    w: [Kw, N] int32, scales: [num_groups, N], zeros: [num_groups, N] int32
    returns float dequantized weight [K, N]
    """
    Kw, N = w.shape
    K = Kw * 8
    num_groups = K // group_size
    assert num_groups == scales.shape[0]

    # unstitch
    w0 = (w & 0xF).to(torch.float32)
    w1 = ((w >> 4) & 0xF).to(torch.float32)
    deq = torch.empty((K, N), device=w.device, dtype=w0.dtype)
    deq[0::8] = w0[::2, :]
    deq[1::8] = w1[::2, :]
    deq[2::8] = (w0[1::2, :] if w0.shape[0] > 1 else w0[0:1, :])
    deq[3::8] = (w1[1::2, :] if w1.shape[0] > 1 else w0[0:1, :])
    # Because each int32 stores 8 int4 weights in four successive rows,
    # split again correctly to rows [2,3] [4,5] [6,7]
    idx = torch.arange(K, device=w.device)[:, None]
    block = idx // 8
    offset_in_block = idx % 8
    gather = ((w[block, :] >> (4 * offset_in_block)) & 0xF).to(torch.float32)
    deq_correct = gather.view(K, N)

    # broadcast scales and zeros
    scales = scales.view(num_groups, 1, N).expand(num_groups, group_size, N).reshape(K, N)
    zeros  = zeros.view(num_groups, 1, N).expand(num_groups, group_size, N).reshape(K, N)
    return deq_correct * scales + zeros


def matmul_dequantize_int4_s2(x: torch.FloatTensor,
                              qweight: torch.IntTensor,
                              scales: torch.FloatTensor,
                              zeros: torch.FloatTensor,
                              group_size: int = 128) -> torch.FloatTensor:
    assert x.is_contiguous()
    assert qweight.is_contiguous()
    M, K = x.shape
    Kw, N = qweight.shape
    assert Kw == K // 8

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
