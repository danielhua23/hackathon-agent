import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_M': 4}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1, 'GROUP_M': 8}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 1, 'GROUP_M': 8}, num_stages=3, num_warps=16), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 4, 'GROUP_M': 4}, num_stages=1, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4, 'GROUP_M': 8}, num_stages=2, num_warps=8)], key=['M', 'N', 'K'], reset_to_zero=['c_ptr'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, scales_ptr, zeros_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_sm, stride_sn, stride_zm, stride_zn, stride_cm, stride_cn, group_size, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K * SPLIT_K)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group * num_pid_n // num_pid_in_group
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k0 = pid_sp_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k0[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k0[:, None] // 8 * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, num_pid_k):
        offs_k = k0 * BLOCK_K * SPLIT_K + offs_k0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
        b_packed = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
        g = offs_k[:, None] // group_size
        scale = tl.load(scales_ptr + g * stride_sm + offs_n[None, :] * stride_sn, mask=offs_n[None, :] < N)
        zero = tl.load(zeros_ptr + g * stride_zm + offs_n[None, :] // 8 * stride_zn, mask=offs_n[None, :] < N)
        b_shift = offs_k[:, None] % 8 * 4
        zp_shift = offs_n[None, :] % 8 * 4
        b_int = b_packed >> b_shift & 15
        zp = zero >> zp_shift & 15
        b = (b_int.astype(tl.float32) - zp.astype(tl.float32)) * scale.astype(tl.float32)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BLOCK_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_K * SPLIT_K // 8 * stride_bk
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, c, mask=mask)
    else:
        tl.store(c_ptrs, c, mask=mask)

def matmul_dequantize_int4_s2(a: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int=128) -> torch.FloatTensor:
    assert a.dtype in (torch.float16, torch.float32)
    assert qweight.dtype == torch.int32
    assert a.is_contiguous()
    assert qweight.is_contiguous()
    assert scales.is_contiguous()
    assert qzeros.is_contiguous()
    device = a.device
    M, K = a.shape
    Kq, N = qweight.shape
    assert K == Kq * 8
    out = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    matmul_kernel[grid](a, qweight, scales, qzeros, out, M, N, K * 1, a.stride(0), a.stride(1), qweight.stride(0), qweight.stride(1), scales.stride(0), scales.stride(1), qzeros.stride(0), qzeros.stride(1), out.stride(0), out.stride(1), group_size)
    return out

def quantize_int4(weight: torch.Tensor, group_size: int=128):
    assert weight.dtype in (torch.float16, torch.float32)
    K, N = weight.shape
    assert K % group_size == 0
    groups = K // group_size
    qweight = torch.empty((K, N // 8), dtype=torch.int32, device=weight.device)
    scales = torch.empty((groups, N), dtype=weight.dtype, device=weight.device)
    zeros = torch.empty((groups, N // 8), dtype=torch.int32, device=weight.device)
    for g in range(groups):
        chunk = weight[g * group_size:(g + 1) * group_size]
        mn = chunk.min(dim=0)[0]
        mx = chunk.max(dim=0)[0]
        scale = ((mx - mn) / 15).clamp(min=1e-08)
        zero = (-mn / scale).round().clamp(0, 15).int()
        q = (chunk / scale + zero).round().clamp(0, 15).int()
        for c in range(0, N, 8):
            col = c // 8
            packed = (q[:, c + 0] | q[:, c + 1] << 4 | q[:, c + 2] << 8 | q[:, c + 3] << 12 | q[:, c + 4] << 16 | q[:, c + 5] << 20 | q[:, c + 6] << 24 | q[:, c + 7] << 28).int()
            qweight[g * group_size:(g + 1) * group_size, col] = packed
            zp_packed = (zero[c + 0] | zero[c + 1] << 4 | zero[c + 2] << 8 | zero[c + 3] << 12 | zero[c + 4] << 16 | zero[c + 5] << 20 | zero[c + 6] << 24 | zero[c + 7] << 28).int()
            zeros[g, col] = zp_packed
        scales[g] = scale
    return (qweight, scales, zeros)

def unpack_int4(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int=128):
    Kq, N8 = qweight.shape
    K = Kq
    N = N8 * 8
    groups = K // group_size
    out = torch.empty((K, N), dtype=scales.dtype, device=qweight.device)
    for g in range(groups):
        g_off = g * group_size
        w_int = torch.empty((group_size, N), dtype=torch.int32, device=qweight.device)
        for c in range(0, N, 8):
            col = c // 8
            packed = qweight[:, col][g * group_size:(g + 1) * group_size, None]
            shift = torch.arange(0, 32, 4, device=qweight.device)[None, :]
            w_int[:, c:c + 8] = packed >> shift & 15
        zp_int = torch.empty(N, dtype=torch.int32, device=qweight.device)
        for c in range(0, N, 8):
            col = c // 8
            packed = zeros[g, col:col + 1]
            shift = torch.arange(0, 32, 4, device=qweight.device)
            zp_int[c:c + 8] = (packed[:, None] >> shift)[0] & 15
        out[g_off:g_off + group_size] = (w_int.float() - zp_int.float()[None, :]) * scales[g:g + 1]
    return out

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
