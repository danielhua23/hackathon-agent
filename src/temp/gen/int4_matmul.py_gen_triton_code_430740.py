
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
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
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
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_idx = k * BLOCK_SIZE_K * SPLIT_K + offs_k[None, :]
        g_idx = k_idx // group_size
        bs_ptrs = bs_ptr + g_idx * stride_bsk + offs_bn[None, :] * stride_bsn
        bzp_ptrs = bzp_ptr + g_idx * stride_bzpk + (offs_bn[None, :] // 8) * stride_bzpn
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        bs = tl.load(bs_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        bzp = tl.load(bzp_ptrs, mask=offs_bn[None, :] < N, other=0)
        b_shift = (offs_k[:, None] % 8) * 4
        z_shift = (offs_n[None, :] % 8) * 4
        b_q = (b >> b_shift) & 0xF
        z_q = (bzp >> z_shift) & 0xF
        b_deq = (b_q.to(tl.float32) - z_q.to(tl.float32)) * bs
        accumulator += tl.dot(a, b_deq.to(a.dtype))
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


def matmul_dequantize_int4_s2(
    x: torch.FloatTensor,
    qweight: torch.IntTensor,
    scales: torch.FloatTensor,
    qzeros: torch.IntTensor,
    group_size: int = 128,
    output: torch.FloatTensor = None
) -> torch.FloatTensor:
    assert x.is_contiguous(), "A must be contiguous"
    assert qweight.is_contiguous(), "qweight must be contiguous"
    M, K = x.shape
    Kq = qweight.shape[0] * 8
    N = qweight.shape[1]
    assert K == Kq, "Leading dimension mismatch"
    assert scales.shape[0] == (K + group_size - 1) // group_size
    assert qzeros.shape[0] == (K + group_size - 1) // group_size
    assert scales.shape[1] == N
    assert qzeros.shape[1] == (N + 7) // 8
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=8),
    ],
    key=['K', 'N'],
)
@triton.jit
def quantize_int4_kernel(
    x_ptr, qweight_ptr, scales_ptr, zeros_packed_ptr,
    K, N,
    stride_xk, stride_xn,
    stride_qw, stride_qwn,
    stride_sc, stride_scn,
    stride_zp, stride_zpn,
    group_size,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    group_pid = tl.program_id(0)
    sub_k = group_pid * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None]
    tid_n = tl.program_id(1)
    sub_n = tid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    mask_k = sub_k < K
    mask_n = sub_n < N
    mask = mask_k & mask_n
    x = tl.load(x_ptr + sub_k * stride_xk + sub_n * stride_xn, mask=mask, other=0.0)
    g_idx = sub_k // group_size
    x_min = tl.min(x, axis=0, keepdim=True)
    x_max = tl.max(x, axis=0, keepdim=True)
    scale = (x_max - x_min) / 15.0
    z = (-x_min / scale).to(tl.int32)
    q = tl.clamp((x.to(tl.float32) / scale + z + 0.5).to(tl.int32), 0, 15)
    q = q.to(tl.int32)
    packed = tl.zeros([BLOCK_SIZE_K, BLOCK_SIZE_N // 8], dtype=tl.int32)
    shifts = tl.arange(0, 8) * 4
    cols_bit = (sub_n % 8) * 4
    q = tl.reshape(q, [BLOCK_SIZE_K, BLOCK_SIZE_N])
    for i in range(0, 8):
        col_i = (sub_n // 8) * 8 + i
        val = tl.where((col_i < N), q[:, col_i], 0)
        shifted = val << (i * 4)
        packed |= shifted
    for i in range(0, 8):
        zp_col = (sub_n // 8) * 8 + i
        shifted_zp = tl.where((zp_col < N), z[:, zp_col], 0) << (i * 4)
        zeros_packed = tl.sum(shifted_zp, axis=1, keepdim=True).to(tl.int32)
    zeros_ptrs = zeros_packed_ptr + g_idx * stride_zpn + (sub_n // 8) * stride_zp
    tl.store(zeros_ptrs, zeros_packed, mask=mask_k)
    qstor = qweight_ptr + (sub_k // 8) * stride_qw + (sub_n // 8) * stride_qwn
    tl.store(qstor, packed, mask=mask_k)
    sc_ptrs = scales_ptr + g_idx * stride_scn + (sub_n) * stride_sc
    tl.store(sc_ptrs, scale, mask=mask_n)


def quantize_int4(x: torch.Tensor, group_size: int = 128):
    x = x.contiguous()
    K, N = x.shape
    qweight = torch.zeros((K // 8, N), dtype=torch.int32, device=x.device)
    scales = torch.empty((K // group_size, N), dtype=torch.float32, device=x.device)
    zeros = torch.empty((K // group_size, (N + 7) // 8), dtype=torch.int32, device=x.device)
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    quantize_int4_kernel[grid](
        x, qweight, scales, zeros,
        K, N,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        group_size
    )
    return qweight, scales, zeros


def unpack_int4(
    qweight: torch.IntTensor,
    scales: torch.FloatTensor,
    zeros: torch.IntTensor,
    group_size: int = 128
) -> torch.FloatTensor:
    Kq, N = qweight.shape
    K = Kq * 8
    weight = torch.empty((K, N), dtype=torch.float32, device=qweight.device)
    zeros_float = torch.empty_like(scales)
    for g in range(scales.shape[0]):
        for n in range(N):
            z = (zeros[g, n // 8] >> ((n % 8) * 4)) & 0xF
            zeros_float[g, n] = float(z)
    for k in range(K):
        for n in range(N):
            q = (qweight[k // 8, n] >> ((k % 8) * 4)) & 0xF
            g = k // group_size
            weight[k, n] = (q - zeros_float[g, n]) * scales[g, n]
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
