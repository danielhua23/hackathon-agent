import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4)], key=['M', 'N', 'K', 'group_size'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_s_g, stride_s_n, stride_z_g, stride_z_n, group_size, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.where(offs_m < M, offs_m, 0)
    offs_n = tl.where(offs_n < N, offs_n, 0)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, k_tiles):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        a_idx = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_idx = b_ptr + (offs_k[:, None] // 8 * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_idx, mask=mask_k[None, :], other=0.0).to(tl.float32)
        packed = tl.load(b_idx, mask=mask_k[:, None], other=0)
        g_id = offs_k // group_size
        s_idx = scales_ptr + g_id * stride_s_g + offs_n[None, :] * stride_s_n
        z_idx = zeros_ptr + g_id * stride_z_g + offs_n[None, :] // 8 * stride_z_n
        s = tl.load(s_idx, mask=mask_k[:, None], other=0.0).to(tl.float32)
        z = tl.load(z_idx, mask=mask_k[:, None], other=0)
        shift = offs_k % 8 * 4
        w4 = packed >> shift[:, None] & 15
        z4 = z >> (offs_n % 8 * 4)[None, :] & 15
        deq = (w4 - z4) * s
        acc += tl.dot(a, deq)
    c_idx = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_idx, acc, mask=c_mask)

def matmul_dequantize_int4_s2(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int) -> torch.Tensor:
    assert x.is_contiguous()
    assert qweight.dtype == torch.int32
    M, K = x.shape
    _, N = scales.shape
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](x, qweight, out, scales, zeros, M, N, K, x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), out.stride(0), out.stride(1), scales.stride(0), scales.stride(1), zeros.stride(0), zeros.stride(1), group_size)
    return out

def quantize_int4(x: torch.Tensor, group_size: int):
    x = x.to(torch.float32).contiguous()
    K, N = x.shape
    assert K % group_size == 0
    full_groups = K // 8
    qweight = torch.empty((full_groups, N), dtype=torch.int32, device=x.device)
    x = x.view(K // group_size, group_size, N)
    max_val = x.max(dim=1, keepdim=True)[0]
    min_val = x.min(dim=1, keepdim=True)[0]
    scale = (max_val - min_val) / 15
    zero = torch.round(-min_val / scale).clamp(0, 15).to(torch.int32)
    xq = torch.round(x / scale + zero).clamp(0, 15).to(torch.int32)
    xq = xq.view(K, N)
    for idx in range(0, N):
        for pack_idx in range(0, full_groups):
            start_k = pack_idx * 8
            end_k = start_k + 8
            block = xq[start_k:end_k, idx]
            packed = block[0] | block[1] << 4 | block[2] << 8 | block[3] << 12 | block[4] << 16 | block[5] << 20 | block[6] << 24 | block[7] << 28
            qweight[pack_idx, idx] = packed
    scale = scale.squeeze(1).transpose(0, 1).contiguous()
    zero = zero.squeeze(1).transpose(0, 1).contiguous()
    zero = zero.view(N, K // group_size)
    return (qweight, scale, zero)

def unpack_int4(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int):
    K_q, N = qweight.shape
    K_full = K_q * 8
    fp = torch.empty((K_full, N), dtype=torch.float32, device=qweight.device)
    scale = scales.transpose(0, 1).contiguous()
    zero = zeros.transpose(0, 1).contiguous()
    for n_idx in range(N):
        for k_pack in range(K_q):
            int32_block = qweight[k_pack, n_idx]
            extracted = torch.empty(8, dtype=torch.int32, device=qweight.device)
            extracted[0] = int32_block >> 0 & 15
            extracted[1] = int32_block >> 4 & 15
            extracted[2] = int32_block >> 8 & 15
            extracted[3] = int32_block >> 12 & 15
            extracted[4] = int32_block >> 16 & 15
            extracted[5] = int32_block >> 20 & 15
            extracted[6] = int32_block >> 24 & 15
            extracted[7] = int32_block >> 28 & 15
            k_start = k_pack * 8
            k_end = k_start + 8
            for i, k_idx in enumerate(range(k_start, k_end)):
                g_idx = k_idx // group_size
                fp[k_idx, n_idx] = (extracted[i] - zero[n_idx, g_idx]) * scale[g_idx, n_idx]
    return fp

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
