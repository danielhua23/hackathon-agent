import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4)], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scale_g, stride_scale_n, stride_zp_g, stride_zp_n, group_size, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    start_k = pid_k * BLOCK_SIZE_K
    offs_k = start_k + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] // 8 * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_curr = k_step * BLOCK_SIZE_K * SPLIT_K + pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_curr[None, :] < K
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b_raw = tl.load(b_ptrs, mask=mask_k, other=0)
        group_idx = k_curr[:, None] // group_size
        shift = k_curr[:, None] % 8 * 4
        bits = b_raw >> shift & 15
        scales = tl.load(scales_ptr + group_idx * stride_scale_g + offs_n[None, :] * stride_scale_n, mask=mask_k, other=0.0)
        zeros = tl.load(zeros_ptr + group_idx * stride_zp_g + offs_n[None, :] // 8 * stride_zp_n, mask=mask_k, other=0.0)
        zeros_bits = zeros >> offs_n[None, :] % 8 * 4 & 15
        b_deq = (bits - zeros_bits) * scales
        accumulator += tl.dot(a, b_deq)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K // 8 * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    out_vals = accumulator.to(c_ptr.dtype.element_ty)
    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, out_vals, mask=mask_c)
    else:
        tl.store(c_ptrs, out_vals, mask=mask_c)

def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.FloatTensor, scales: torch.FloatTensor, zeros: torch.FloatTensor, split_k: int=1) -> torch.FloatTensor:
    assert x.dim() == 2 and qweight.dim() == 2 and (scales.dim() == 2) and (zeros.dim() == 2)
    M, K = x.shape
    assert K == qweight.shape[0] * 8
    N = qweight.shape[1]
    assert scales.shape == (K // scales.shape[0], N)
    assert zeros.shape == (K // zeros.shape[0], N)
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), split_k)
    matmul_kernel[grid](x, qweight, output, scales, zeros, M, N, K, x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), scales.stride(1), zeros.stride(0), zeros.stride(1), scales.shape[0])
    return output

def quantize_int4(w: torch.Tensor, group_size: int=128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    w: [OC, K] (fp16/fp32 weights)
    returns: (qpacked, scales, zerospacked, group_size)
       qpacked : int32, shape [OC, K//8]
       scales  : fp16/fp32 [OC, K//group_size]
       zerospacked : int32 [K//group_size, OC//8]
    """
    assert w.dim() == 2
    OC, K = w.shape
    assert K % group_size == 0
    w = w.view(-1, K)
    OC_total, K_ = w.shape
    groups_per_row = K_ // group_size
    w = w.view(OC_total, groups_per_row, group_size)
    w_min = w.min(dim=2, keepdim=True).values
    w_max = w.max(dim=2, keepdim=True).values
    scales = (w_max - w_min) / 15.0
    zeros = torch.round(-w_min / scales).clamp(0, 15).to(torch.int32)
    quant = torch.clamp(torch.round(w / scales + zeros), 0, 15).to(torch.int32)
    OC_pack = OC_total // 8
    zeros_pack = zeros.permute(1, 0, 2).contiguous().view(groups_per_row, OC_pack, 8)
    zeros_packed = zeros_pack[..., 0]
    for p in range(1, 8):
        zeros_packed |= zeros_pack[..., p] << p * 4
    zeros_packed = zeros_packed.view(groups_per_row, OC_pack).contiguous()
    K_pack = K_ // 8
    quant = quant.view(OC_total, K_)
    qpacked = torch.empty((OC_total, K_pack), dtype=torch.int32, device=w.device)
    for p in range(8):
        qpacked |= quant[:, p::8] << p * 4
    qpacked = qpacked.contiguous()
    scales = scales.view(OC_total, groups_per_row).contiguous()
    return (qpacked, scales, zeros_packed, group_size)

def unpack_int4(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int=128) -> torch.Tensor:
    """
    qweight : int32 [OC, K//8]
    scales  : [OC, K//group_size]
    zeros   : int32 [K//group_size, OC//8]
    returns : fp16/fp32 tensor [OC, K]
    """
    OC, K8 = qweight.shape
    K = K8 * 8
    group_dim = K // group_size
    assert scales.shape == (OC, group_dim)
    assert zeros.shape == (group_dim, OC // 8)
    quant = torch.empty((OC, K), dtype=torch.int32, device=qweight.device)
    for p in range(8):
        mask = 15 << p * 4
        quant[:, p::8] = (qweight & mask) >> p * 4
    OC8 = OC // 8
    zeros_ext = torch.empty((group_dim, OC), dtype=torch.int32, device=zeros.device)
    for p in range(8):
        mask = 15 << p * 4
        zeros_ext[:, p::8] = (zeros & mask) >> p * 4
    zeros_ext = zeros_ext.permute(1, 0).contiguous()
    scales_mat = scales.view(OC, group_dim).unsqueeze(-1).expand(-1, -1, group_size).reshape(OC, K)
    zeros_mat = zeros_ext.view(OC, group_dim).unsqueeze(-1).expand(-1, -1, group_size).reshape(OC, K)
    return (quant.float() - zeros_mat) * scales_mat

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
