import torch
import triton
import triton.language as tl

@triton.jit
def _pack_int4x2(x0, x1):
    x0 = x0 & 15
    x1 = x1 & 15
    return x1 | x0 << 4

@triton.jit
def _unpack_int4x2(b):
    low = b >> 0 & 15
    high = b >> 4 & 15
    return (low, high)

@triton.jit
def matmul_kernel(q_ptr, a_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_a_m, stride_a_k, stride_q_k2, stride_q_n, stride_s_g, stride_s_n, stride_z_g, stride_z_n, stride_c_m, stride_c_n, group_size, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K2: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_k = tl.cdiv(K, BLOCK_K2 * 2)
    grid_m = pid // num_pid_n
    grid_n = pid % num_pid_n
    rm = grid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = grid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk_packed = pid_z * BLOCK_K2 + tl.arange(0, BLOCK_K2)
    rk = rk_packed * 2
    a_ptrs = a_ptr + rm[:, None] * stride_a_m + rk[None, :] * stride_a_k
    a_mask = (rm[:, None] < M) & (rk[None, :] < K)
    A = tl.load(a_ptrs, mask=a_mask, other=0.0)
    q_ptrs = q_ptr + (rk_packed[:, None] * stride_q_k2 + rn[None, :] * stride_q_n)
    packed = tl.load(q_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0)
    lo, hi = _unpack_int4x2(packed)
    q_vals = tl.interleave(lo, hi)
    group_idx = rk[:, None] // group_size
    s_ptrs = scales_ptr + group_idx * stride_s_g + rn[None, :] * stride_s_n
    z_ptrs = zeros_ptr + group_idx * stride_z_g + rn[None, :] * stride_z_n
    scales = tl.load(s_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
    zeros = tl.load(z_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
    b_fp = (q_vals - zeros) * scales
    acc = tl.dot(A, b_fp)
    c_offs = c_ptr + rm[:, None] * stride_c_m + rn[None, :] * stride_c_n
    c_mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K > 1:
        tl.atomic_add(c_offs, acc, mask=c_mask)
    else:
        tl.store(c_offs, acc, mask=c_mask)

@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K2': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K2': 32, 'SPLIT_K': 2}, num_stages=4, num_warps=8)], key=['M', 'N', 'K'])
@triton.jit
def matmul_dequantize_int4_s2(q_ptr, a_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_a_m, stride_a_k, stride_q_k2, stride_q_n, stride_s_g, stride_s_n, stride_z_g, stride_z_n, stride_c_m, stride_c_n, group_size, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K2: tl.constexpr, SPLIT_K: tl.constexpr):
    matmul_kernel(q_ptr, a_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_a_m, stride_a_k, stride_q_k2, stride_q_n, stride_s_g, stride_s_n, stride_z_g, stride_z_n, stride_c_m, stride_c_n, group_size, BLOCK_M, BLOCK_N, BLOCK_K2, SPLIT_K)

def matmul_dequantize_int4_s2(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int=128) -> torch.Tensor:
    """
    Python-launchable GEMM with INT4 quantized weights.

    Memory layout expected:
        qweight   – [ K//2 , N ]   int32   column-major
        scales    – [ G , N ]      float   column-major,  G = K // group_size
        qzeros    – same shape as scales
        x         – [ M , K ]      fp16/fp32 row-major
    Output:
        c         – [ M , N ]      fp32
    """
    x = x.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    qzeros = qzeros.contiguous()
    M, K = x.shape
    N = qweight.size(1)
    device = x.device
    out = torch.empty((M, N), dtype=torch.float32, device=device)

    def grid(meta):
        grid_m = triton.cdiv(M, meta['BLOCK_M'])
        grid_n = triton.cdiv(N, meta['BLOCK_N'])
        grid_z = meta['SPLIT_K']
        return (grid_m * grid_n, grid_z)
    matmul_dequantize_int4_s2[grid](qweight, x, out, scales, qzeros, M, N, K, x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1), scales.stride(0), scales.stride(1), qzeros.stride(0), qzeros.stride(1), out.stride(0), out.stride(1), group_size)
    return out

def quantize_int4(x: torch.Tensor, group_size: int=128, transpose: bool=True):
    """
    Quantise weight matrix (row-vector row-major) into INT4.

    Returns tensors that are column-major (as expected by Triton kernel).

    Args
    ----
    x : [K_orig, N] float
    Returns
    -------
    packed   : [ K_orig//2 , N ] int32  column-major
    scales   : [ G , N ] float           column-major, G = K_orig//group_size
    zeros    : [ G , N ] float           column-major
    """
    K_orig, N = x.shape
    assert K_orig % group_size == 0
    G = K_orig // group_size
    x = x.view(G, group_size, N)
    x_min = x.min(dim=1, keepdim=True)[0]
    x_max = x.max(dim=1, keepdim=True)[0]
    scales = (x_max - x_min) / 15.0
    zeros = torch.round(-x_min / scales).clamp(0, 15)
    q = torch.round(x / scales + zeros).clamp(0, 15).to(torch.uint8)
    q = q.view(K_orig, N)
    if transpose:
        q = q.T.contiguous()
        scales = scales.squeeze(1).T.contiguous()
        zeros = zeros.squeeze(1).T.contiguous()
    else:
        scales = scales.squeeze(1).contiguous()
        zeros = zeros.squeeze(1).contiguous()
    packed = torch.zeros((N, K_orig // 2), dtype=torch.int32, device=x.device)
    for k in range(0, K_orig, 2):
        low = q[:, k].to(torch.int32)
        high = q[:, k + 1].to(torch.int32)
        packed[:, k // 2] = (high & 15) << 4 | low & 15
    return (packed.view(K_orig // 2, N).contiguous(), scales, zeros)

def unpack_int4(packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int=128):
    """
    De-quantize the output of quantize_int4 back to float for validation.
    Assumes column-major layout (same as the kernel).
    Returns tensor of shape [K, N] – float32, column-major
    """
    packed = packed.contiguous()
    scales = scales.contiguous()
    zeros = zeros.contiguous()
    K2, N = packed.shape
    K = K2 * 2
    device = packed.device
    unpacked = torch.zeros((K, N), dtype=torch.float32, device=device)
    for col in range(N):
        pack = packed[:, col].clone()
        even = pack & 15
        odd = pack >> 4 & 15
        int_vec = torch.empty(K, dtype=torch.float32, device=device)
        int_vec[0::2] = even.float()
        int_vec[1::2] = odd.float()
        group_idx = torch.arange(K, device=device) // group_size
        s = scales[group_idx, col]
        z = zeros[group_idx, col]
        unpacked[:, col] = (int_vec - z) * s
    return unpacked

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
