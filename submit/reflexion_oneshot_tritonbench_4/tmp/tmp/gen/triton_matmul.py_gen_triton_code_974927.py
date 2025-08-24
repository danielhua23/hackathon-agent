import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=1, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=1, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=16)], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group % num_pid_n
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    C_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_loop = tl.cdiv(K, BLOCK_K)
    for k_idx in tl.static_range(8):
        if k_idx < k_loop:
            a = tl.load(A_ptrs, mask=None, other=0.0)
            b = tl.load(B_ptrs, mask=None, other=0.0)
        else:
            a = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            b = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        accumulator += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    mask_m = offs_am[:, None] < M
    mask_n = offs_bn[None, :] < N
    tl.store(C_ptrs, accumulator, mask=mask_m & mask_n)

def matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor=None, eps: float=1e-06) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype
    assert a.device == b.device
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    if out is None:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    else:
        assert out.shape == (M, N) and out.dtype == a.dtype and out.is_contiguous()
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](a, b, out, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1))
    return out

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