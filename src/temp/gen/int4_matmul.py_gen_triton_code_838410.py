
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, scales, zeros,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  stride_scales_g, stride_scales_n,
                  stride_zeros_g, stride_zeros_n,
                  groupsize,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  SPLIT_K: tl.constexpr = 1,
                  GROUP_SIZE_M: tl.constexpr = 8):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
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
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_idx = tl.arange(0, BLOCK_SIZE_K)
    group_id_k = k_idx // groupsize
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + ((offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn)
    scales_ptrs = scales + group_id_k[None, :] * stride_scales_g + offs_n[None, :] * stride_scales_n
    zeros_ptrs = zeros + group_id_k[None, :] * stride_zeros_g + offs_n[None, :] * stride_zeros_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_idx = pos_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        group_id_k = pos_k // groupsize
        scales_ptrs = scales + group_id_k[None, :] * stride_scales_g + offs_n[None, :] * stride_scales_n
        zeros_ptrs = zeros + group_id_k[None, :] * stride_zeros_g + offs_n[None, :] * stride_zeros_n

        mask_k = pos_k < K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        
        b_idx = pos_k // 8
        b = tl.load(B + b_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=mask_k[:, None], other=0)
        
        scales = tl.load(scales_ptrs, mask=mask_k[None, :], other=0.0)
        zeros = tl.load(zeros_ptrs, mask=mask_k[None, :], other=0.0)

        vec = tl.arange(0, 8)
        shift = (pos_k % 8) * 4
        weights = (b >> shift[:, None]) & 0xF
        
        b_f = (weights - zeros) * scales
        
        accumulator += tl.dot(a, b_f)

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cm = offs_cm < M
    mask_cn = offs_cn < N
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = mask_cm[:, None] & mask_cn[None, :]

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=mask)

configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
]

@triton.autotune(configs=configs, key=["M", "N", "K"], use_cuda_graph=False)
@triton.jit
def matmul_dequantize_int4_s2(
    A, B, C, scales, zeros,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    groupsize,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr = 1
):
    matmul_kernel(
        A, B, C, scales, zeros,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_scales_g, stride_scales_n,
        stride_zeros_g, stride_zeros_n,
        groupsize,
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_N=BLOCK_SIZE_N, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SPLIT_K=SPLIT_K
    )

def quantize_int4(x: torch.Tensor, groupsize: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.t()
    W = x
    M, N = W.shape[0], W.shape[1]
    W = W.reshape((M, N))

    groups = M // groupsize
    
    W = W.reshape((groups, -1, N))
    Wmin = W.min(dim=1, keepdim=True)[0]
    Wmax = W.max(dim=1, keepdim=True)[0]
    
    scale = (Wmax - Wmin) / 15
    zero = -Wmin / scale
    
    scale = scale.expand(groups, groupsize, N).reshape(M, -1)
    zero = zero.expand(groups, groupsize, N).reshape(M, -1)
    
    int_weight = torch.clamp(torch.round(W / scale.unsqueeze(1) + zero.unsqueeze(1)), 0, 15)
    
    intweight = int_weight.reshape(M, N)
    
    packed_weight = torch.zeros((W.shape[0], W.shape[2] // 2), dtype=torch.int32, device=x.device)
    for col in range(0, intweight.shape[1], 8):
        v = intweight[:, col:col+8].contiguous().view(torch.int8)
        v = v & 0xF  # ensure 4bit
        v1 = v[:, ::2] & 0xF
        v2 = (v[:, 1::2] & 0xF) << 4
        packed_weight[:, col//2] = (v1 | v2).view(torch.int32).squeeze()

    packed_weight = packed_weight.t()
    
    scale = scale[::groupsize, :].t()
    zero = zero[::groupsize, :].t()
    
    return packed_weight, scale, zero

def unpack_int4(
    packed_weights: torch.Tensor, 
    scales: torch.Tensor, 
    zeros: torch.Tensor, 
    groupsize: int
) -> torch.Tensor:
    N, M = packed_weights.shape
    unpacked = torch.zeros((N, M * 8), dtype=torch.float16, device=packed_weights.device)
    
    for i in range(M):
        chunk = packed_weights[:, i]
        for j in range(8):
            lower = (chunk >> (j * 4)) & 0xF
            unpacked[:, i * 8 + j] = lower.float()
    
    scales = scales.repeat_interleave(groupsize, dim=0)
    zeros = zeros.repeat_interleave(groupsize, dim=0)
    
    unpacked = (unpacked - zeros[:N, :M*8]) * scales[:N, :M*8]
    return unpacked.t()


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
