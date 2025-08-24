import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=2, num_warps=4)], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        acc += tl.dot(a.to(tl.float16), b.to(tl.float16))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, acc.to(a_ptr.dtype.element_ty), mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == b.dim() == 2
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions for GEMM'
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert a.dtype in (torch.float16, torch.float32)
    assert b.dtype == a.dtype
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c

##################################################################################################################################################





import torch



# Test for matmul

def test_matmul():

    results = {}

    M, K, N = 256, 128, 256



    # Test case 1: torch.float16

    a = torch.randn((M, K), dtype=torch.float16, device='cuda')

    b = torch.randn((K, N), dtype=torch.float16, device='cuda')

    c = matmul(a, b)

    results['test_case_1'] = c



    return results



# Run all tests

result_gold = test_matmul()