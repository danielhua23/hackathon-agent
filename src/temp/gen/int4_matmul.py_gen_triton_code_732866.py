
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
        k_idx = k * BLOCK_SIZE_K * SPLIT_K + offs_k[None, :]
        g_idx = k_idx // group_size
        bs_ptrs = bs_ptr + g_idx * stride_bsk + offs_n[None, :] * stride_bsn
        bzp_ptrs = bzp_ptr + g_idx * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
        bs = tl.load(bs_ptrs, mask=offs_n[None, :] < N, other=0.0)
        bzp = tl.load(bzp_ptrs, mask=offs_n[None, :] < N, other=0)
        b_shift = (offs_k[:, None] % 8) * 4
        z_shift = (offs_n[None, :] % 8) * 4
        b_q = (b >> b_shift) & 0xF
        z_q = (bzp >> z_shift) & 0xF
        b_deq = ((b_q.to(tl.float32) - z_q.to(tl.float32)) * bs).to(a.dtype)
        accumulator += tl.dot(a, b_deq)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int = 128, output: torch.FloatTensor = None) -> torch.FloatTensor:
    assert x.is_contiguous(), "input must be contiguous"
    assert qweight.is_contiguous(), "qweight must be contiguous"
    M, K = x.shape
    Kq = qweight.shape[0] * 8
    N = qweight.shape[1]
    assert K == Kq, "Leading dimension of A must match unpacked columns of quantized B"
    assert scales.shape[0] == (K + group_size - 1) // group_size, "Scales shape along rows invalid"
    assert qzeros.shape[0] == (K + group_size - 1) // group_size, "Qzeros shape along rows invalid"
    assert scales.shape[1] == N, "Scales shape along cols invalid"
    assert qzeros.shape[1] == (N + 7) // 8 * 8, "Qzeros shape along cols invalid"
    if output is None:
        output = torch.zeros((M, N), device=x.device, dtype=x.dtype)
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


@triton.jit
def quantize_int4_kernel(
    x_ptr, qweight_ptr, scales_ptr, zeros_ptr,
    N, K,
    stride_xn, stride_xk,
    stride_qw, stride_qwn,
    stride_sc, stride_scn,
    stride_zp, stride_zpn,
    group_size,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    nk = tl.program_id(0)
    nk_k = nk % (K // BLOCK_SIZE_K)
    nk_n = nk // (K // BLOCK_SIZE_K)
    offs_k = nk_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = nk_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    mask_k = offs_k < K
    mask = mask_n[:, None] & mask_k[None, :]

    x_ptrs = x_ptr + offs_n[:, None] * stride_xn + offs_k[None, :] * stride_xk
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    g_idx = offs_k[None, :] // group_size
    x_min = tl.min(x, axis=1, keepdim=True)
    x_max = tl.max(x, axis=1, keepdim=True)
    scale = (x_max - x_min) / 15.0
    zero = (-x_min / scale).to(tl.int32)
    q = tl.clamp((x.to(tl.float32) / scale + zero + 0.5).to(tl.int32), 0, 15)

    scale = tl.reshape(scale, [BLOCK_SIZE_N])
    zero = tl.reshape(zero, [BLOCK_SIZE_N])

    packed = tl.zeros([BLOCK_SIZE_N], dtype=tl.int32)
    for i in range(0, 8):
        off = offs_k[i::8]
        cols = tl.arange(0, BLOCK_SIZE_N)[:, None]
        q_i = q[cols, off[None, :]]
        packed |= (q_i & 0xF) << (i * 4)

    qweight_ptrs = qweight_ptr + offs_n * stride_qw + nk_k * stride_qwn
    scales_ptrs = scales_ptr + offs_n * stride_sc + g_idx[0, 0] * stride_scn
    zeros_ptrs = zeros_ptr + (offs_n // 8) * stride_zp + (nk_k * 8 + offs_k[0]) // group_size * stride_zpn

    tl.store(qweight_ptrs, packed, mask=mask_n)
    tl.store(scales_ptrs, scale, mask=mask_n)
    tl.store(zeros_ptrs, packed, mask=mask_n)   # placeholder
    zeros = tl.reshape(zero, [BLOCK_SIZE_N])
    tl.store(zeros_ptrs, zeros, mask=mask_n)


def quantize_int4(x: torch.Tensor, group_size: int = 128):
    x = x.contiguous().float()
    K, N = x.shape
    assert K % group_size == 0, "K must be divisible by group_size"
    packed = torch.zeros((K // 8, N), dtype=torch.int32, device=x.device)
    scales = torch.empty((K // group_size, N), dtype=torch.float32, device=x.device)
    zeros = torch.empty((K // group_size, (N + 7) // 8), dtype=torch.int32, device=x.device)

    x_float = x.clone()
    xq = torch.zeros_like(x_float)
    zeros_float = torch.zeros((K // group_size, N), device=x.device)
    for g in range(0, K // group_size):
        xs = x_float[g * group_size:(g + 1) * group_size, :]
        x_min = xs.min(dim=0, keepdim=True)[0]
        x_max = xs.max(dim=0, keepdim=True)[0]
        scale = (x_max - x_min) / 15.0
        zero = (-x_min / scale).round().clamp(0, 15)
        xq[g * group_size:(g + 1) * group_size, :] = (xs / scale + zero).round().clamp(0, 15)
        scales[g, :] = scale.squeeze(0)
        zeros_float[g, :] = zero.squeeze(0)
    for n in range(N):
        for g in range(K // group_size):
            zeros[g, n // 8] |= (zeros_float[g, n].int() << ((n % 8) * 4))
    for k in range(0, K, 8):
        for n in range(N):
            for i in range(8):
                packed[k // 8, n] |= (xq[k + i, n].int() << (i * 4))
    return packed, scales, zeros


def unpack_int4(qweight: torch.IntTensor, scales: torch.FloatTensor, zeros: torch.FloatTensor, group_size: int = 128) -> torch.FloatTensor:
    Kq, N = qweight.shape
    K = Kq * 8
    weight = torch.zeros((K, N), dtype=torch.float32, device=qweight.device)

    zeros_float = torch.zeros((scales.shape[0], N), dtype=torch.float32, device=qweight.device)
    for g in range(zeros.shape[0]):
        for n in range(N):
            z = (zeros[g, n // 8] >> ((n % 8) * 4)) & 0xF
            zeros_float[g, n] = float(z)
    for k in range(K):
        for n in range(N):
            q = (qweight[k // 8, n] >> ((k % 8) * 4)) & 0xF
            g = k // group_size
            scale = scales[g, n]
            zero = zeros_float[g, n]
            weight[k, n] = (float(q) - zero) * scale
    return weight.contiguous()


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
