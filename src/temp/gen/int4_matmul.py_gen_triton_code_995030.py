
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
                       'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
                       'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2}, num_stages=2, num_warps=4),
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

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k_step = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    k_step = BLOCK_SIZE_K * SPLIT_K
    k_last = min((pid_k + 1) * BLOCK_SIZE_K, K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_step[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k_step[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, k_step)):
        k_off = k * k_step
        a_mask = (offs_k_step[None, :] + k_off < K) & (offs_m[:, None] < M)
        b_mask = (offs_k_step[:, None] + k_off < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        packed = tl.load(b_ptrs, mask=b_mask, other=0)

        gidx = ((offs_k_step[:, None] + k_off) // group_size)[:, 0]
        gidx = tl.view(gidx, (BLOCK_SIZE_K, 1))
        scales = tl.load(scales_ptr + gidx * stride_bsk + offs_n[None, :] * stride_bsn,
                         mask=b_mask, other=0)

        bzp = tl.load(zeros_ptr + gidx * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn,
                      mask=b_mask, other=0)

        shift = ((offs_k_step[:, None] + k_off) % 8) * 4
        int_b = ((packed >> shift) & 0xF).to(tl.float32)

        zp_shift = (offs_n[None, :] % 8) * 4
        int_zp = ((bzp >> zp_shift) & 0xF).to(tl.float32)

        b = (int_b - int_zp) * scales
        acc += tl.dot(a, b)

        a_ptrs += k_step * stride_ak
        b_ptrs += (k_step // 8) * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask_c)
    else:
        tl.atomic_add(c_ptrs, c, mask=mask_c)

# =========================
# Front-end helpers
# =========================
def quantize_int4(weights: torch.Tensor, group_size: int = 128) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weights.dim() == 2
    K, N = weights.shape
    assert K % group_size == 0

    num_groups = K // group_size
    w_groups = weights.view(num_groups, group_size, N)
    wmin, wmax = w_groups.aminmax(dim=1)
    scale = (wmax - wmin) / 15.0
    scale = torch.where(scale == 0,
                        torch.tensor(1.0, dtype=scale.dtype, device=scale.device),
                        scale)
    zero_fp = -wmin / scale
    q = ((w_groups / scale.unsqueeze(1) + zero_fp.unsqueeze(1) + 0.5)
         .floor().clamp(0, 15).to(torch.int32))

    q = q.view(K, N)
    packed = torch.empty((K // 8, N), dtype=torch.int32, device=weights.device)
    for k in range(0, 8):
        packed |= (q[k::8] & 0xF) << (k * 4)

    zero_int = zero_fp.round().int().clamp(0, 15)
    zeros_packed = torch.empty((num_groups, N // 8), dtype=torch.int32, device=weights.device)
    for n8 in range(0, 8):
        zeros_packed |= ((zero_int.view(num_groups * N)[n8::8] & 0xF)
                         << (n8 * 4))
    zeros_packed = zeros_packed.view(num_groups, N // 8)

    return packed, scale, zeros_packed


def unpack_int4(w, scales, zeros, group_size: int = 128):
    K = w.shape[0] * 8
    N = w.shape[1]
    assert w.ndim == 2 and scales.ndim == 2 and zeros.ndim == 2
    num_groups = scales.size(0)

    deq = torch.zeros((K, N), dtype=torch.float32, device=w.device)
    for k in range(K):
        for n in range(N):
            k_block = k // 8
            k_nibble = k % 8
            val = (w[k_block, n] >> (k_nibble * 4)) & 0xF
            group = k // group_size
            gp_n = n // 8
            znib = n % 8
            zp = (zeros[group, gp_n] >> (znib * 4)) & 0xF
            deq[k, n] = (float(val) - float(zp)) * scales[group, n]
    return deq


def matmul_dequantize_int4_s2(x: torch.FloatTensor,
                              qweight: torch.IntTensor,
                              scales: torch.FloatTensor,
                              zeros: torch.IntTensor,
                              group_size: int = 128) -> torch.FloatTensor:
    assert x.is_contiguous()
    assert qweight.is_contiguous()
    M, K = x.shape
    Kw, N = qweight.shape
    assert K == Kw * 8

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
