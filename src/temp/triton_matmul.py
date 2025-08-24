
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    IS_EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc_dtype = tl.float32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    mask_k = offs_k
    for k in range(0, K, BLOCK_SIZE_K):
        if IS_EVEN_K:
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (mask_k[None, :] < (K - k)), other=0.0)
            b = tl.load(b_ptrs, mask=(mask_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b).to(acc_dtype)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.type.element_ty), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "a must be contiguous"
    assert b.is_contiguous(), "b must be contiguous"
    M, K = a.shape
    _K, N = b.shape
    assert K == _K, "K dimensions must match"

    def decide_dtype(dt):
        if dt in (torch.float8_e4m3fn, torch.float8_e5m2):
            return torch.float16
        elif dt in (torch.float16, torch.bfloat16, torch.float32):
            return torch.float16
        elif dt is torch.int8:
            return torch.int32
        else:
            return torch.float32

    c_dtype = decide_dtype(a.dtype)
    c = torch.empty((M, N), dtype=c_dtype, device=a.device)

    if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 128
        num_stages, num_warps = 3, 8
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        num_stages, num_warps = 2, 4

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        IS_EVEN_K=K % BLOCK_K == 0,
        num_stages=num_stages,
        num_warps=num_warps,
    )
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