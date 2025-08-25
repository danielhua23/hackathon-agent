import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4)], key=['M', 'N', 'K'], reset_to_zero=['c_ptr'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, bs_ptr, bzp_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_bsk, stride_bsn, stride_bzpk, stride_bzpn, group_size, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] // 8 * stride_bk + offs_bn[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        bs_ptrs = bs_ptr + (offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size * stride_bsk + offs_bn[None, :] * stride_bsn
        bzp_ptrs = bzp_ptr + (offs_k[:, None] + k * BLOCK_SIZE_K * SPLIT_K) // group_size * stride_bzpk + offs_bn[None, :] // 8 * stride_bzpn
        b_shift_bits = offs_k[:, None] % 8 * 4
        bzp_shift_bits = offs_bn[None, :] % 8 * 4
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        bs = tl.load(bs_ptrs)
        bzp = tl.load(bzp_ptrs)
        int_b = b >> b_shift_bits & 15
        int_bzp = bzp >> bzp_shift_bits & 15
        b = ((int_b - int_bzp) * bs).to(a.dtype)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk // 8
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)

def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int=128, output=None) -> torch.FloatTensor:
    assert x.is_contiguous(), 'A must be contiguous'
    assert qweight.is_contiguous(), 'B must be contiguous'
    M, K = x.shape
    N = scales.shape[1]
    if output is None:
        output = torch.zeros((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META['SPLIT_K'])
    matmul_kernel[grid](x, qweight, output, scales, qzeros, M, N, K, x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), scales.stride(1), qzeros.stride(0), qzeros.stride(1), group_size)
    return output

def quantize_int4(weight, group_size=128, tp_rank=0):
    weight = weight.transpose(1, 0)
    h1, h2 = weight.shape
    assert h1 % 8 == 0 and h2 % 8 == 0, 'H1 {} H2 {}'.format(h1, h2)
    assert h2 % group_size == 0, 'H1 {} H2 {}'.format(h1, h2)
    weight = weight.contiguous().view(-1, group_size).cuda(tp_rank)
    weight_max = weight.amax(-1, keepdim=True)
    weight_max = torch.where(weight_max < 0, 0, weight_max)
    weight_min = weight.amin(-1, keepdim=True)
    weight_min = torch.where(weight_min > 0, 0, weight_min)
    weight_range = weight_max - weight_min
    scale = weight_range / (2 ** 4 - 1)
    zero_point = (-weight_min / scale).round().clamp(0, 15).to(torch.int32)
    weight = (weight / scale + zero_point).round().clamp(0, 15).to(torch.int32).view(h1, h2)
    int_weight = torch.empty(h1, h2 // 8, dtype=torch.int32, device=weight.device)
    int_zero_point = torch.zeros(h1 // 8, h2 // group_size, dtype=torch.int32, device=weight.device)
    zero_point = zero_point.view(h1, -1)
    scale = scale.view(h1, -1)
    for pack in range(0, h2, 8):
        for i in range(8):
            int_weight[:, pack // 8] += weight[:, pack + i] << i * 4
    for pack in range(0, h1, 8):
        for i in range(8):
            int_zero_point[pack // 8, :] += zero_point[pack + i, :] << i * 4
    return (int_weight.transpose(1, 0).contiguous(), scale.transpose(1, 0).contiguous(), int_zero_point.transpose(1, 0).contiguous(), group_size)

def unpack_int4(weight, scale, zp):
    weight = weight.transpose(1, 0)
    scale = scale.transpose(1, 0)
    zp = zp.transpose(1, 0)
    h1, h2 = weight.shape
    group_size = h2 * 8 // scale.shape[1]
    group_num = scale.shape[1]
    fp_weight = torch.zeros(h1, h2 * 8, dtype=torch.half, device=weight.device)
    fp_zero_point = torch.zeros(h1, group_num, device=weight.device)
    for pack in range(0, h2):
        for i in range(8):
            fp_weight[:, pack * 8 + i] = weight[:, pack] >> i * 4 & 15
    for pack in range(0, h1 // 8):
        for i in range(8):
            fp_zero_point[pack * 8 + i, :] = zp[pack, :] >> i * 4 & 15
    for g in range(group_num):
        fp_weight[:, g * group_size:(g + 1) * group_size] = (fp_weight[:, g * group_size:(g + 1) * group_size] - fp_zero_point[:, g].unsqueeze(1)) * scale[:, g].unsqueeze(1)
    return fp_weight.transpose(1, 0)

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
