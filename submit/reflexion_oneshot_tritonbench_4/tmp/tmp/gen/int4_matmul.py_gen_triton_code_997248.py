import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4)], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales, stride_zeros, group_size, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr=1):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    offs_kk = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am * stride_am + offs_kk[None, :] * stride_ak
    b_ptrs = b_ptr + offs_kk[:, None] // 8 * stride_bk + offs_bn * stride_bn
    group_idx = offs_kk[:, None] // group_size
    scales_ptrs = scales_ptr + group_idx * stride_scales + offs_bn
    zeros_ptrs = zeros_ptr + group_idx * stride_zeros + offs_bn // 8
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    base_k = 0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=(offs_am < M) & (offs_kk[None, :] < K - base_k), other=0.0)
        b_i32 = tl.load(b_ptrs, mask=(offs_kk[:, None] < K - base_k) & (offs_bn < N), other=0)
        shift = (base_k + offs_kk[:, None]) % 8 * 4
        b_4b = b_i32 >> shift & 15
        scale = tl.load(scales_ptrs, mask=offs_kk[:, None] < K - base_k, other=1.0)
        zero = tl.load(zeros_ptrs, mask=(offs_kk[:, None] < K - base_k) & (offs_bn < N), other=0)
        zero = zero >> offs_bn % 8 * 4 & 15
        b_deq = (b_4b.to(tl.float32) - zero.to(tl.float32)) * scale
        accumulator += tl.dot(a, b_deq, out_dtype=tl.float32)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K // 8 * stride_bk
        base_k += BLOCK_SIZE_K * SPLIT_K
        scales_ptrs += BLOCK_SIZE_K * SPLIT_K // group_size * stride_scales
        zeros_ptrs += BLOCK_SIZE_K * SPLIT_K // group_size * stride_zeros
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    c_ptrs = c_ptr + offs_cm * stride_cm + offs_cn * stride_cn
    mask = (offs_cm < M) & (offs_cn < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=mask)

def matmul_dequantize_int4_s2(a, qweight, scales, zeros, group_size=128, output=None, split_k=1):
    assert a.shape[-1] == qweight.shape[0] * 8
    assert qweight.shape[-1] == scales.shape[-1]
    assert a.dtype == scales.dtype == zeros.dtype
    assert a.device == qweight.device == scales.device == zeros.device
    M, K = a.shape
    N = qweight.shape[1]
    if output is None:
        output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), split_k)
    matmul_kernel[grid](a, qweight, output, scales, zeros, M, N, K, stride_am=a.stride(0), stride_ak=a.stride(1), stride_bk=qweight.stride(0), stride_bn=qweight.stride(1), stride_cm=output.stride(0), stride_cn=output.stride(1), stride_scales=scales.stride(0), stride_zeros=zeros.stride(0), group_size=group_size)
    return output

def quantize_int4(weights, group_size=128):
    N, K = weights.shape
    assert K % group_size == 0
    n_groups = K // group_size
    w = weights.view(N, n_groups, group_size)
    wmin = w.amin(dim=-1, keepdim=True)
    wmax = w.amax(dim=-1, keepdim=True)
    scales = (wmax - wmin) / 15.0
    zeros = torch.clamp(-wmin / scales, max=15).round()
    q = torch.clamp(torch.round(w / scales + zeros), 0, 15).to(torch.uint8)
    q = q.view(N, K)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=weights.device)
    for i in range(8):
        packed |= (q.view(-1, 8)[:, i].view(N, K // 8) & 15) << 4 * i
    return (packed.T.contiguous(), scales.squeeze(-1).contiguous().T, zeros.squeeze(-1).contiguous().T, None)

def unpack_int4(q, scales, zeros, group_size=128):
    q = q.T
    scales = scales.T
    zeros = zeros.T
    N, K8 = q.shape
    K = K8 * 8
    unpacked = torch.empty((N, K), dtype=scales.dtype, device=q.device)
    for i in range(8):
        idx = torch.arange(i, K, 8, device=q.device)
        row_idx = idx // 8
        bit = q[:, idx // 8] >> i * 4 & 15
        grp = idx // group_size
        unpacked[:, idx] = (bit - zeros[:, grp]) * scales[:, grp]
    return unpacked.contiguous()

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
