\n\nimport torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef matmul_kernel(\n    a_ptr, b_ptr, c_ptr,\n    M, N, K,\n    stride_am, stride_ak,\n    stride_bk, stride_bn,\n    stride_cm, stride_cn,\n    BLOCK_M: tl.constexpr,\n    BLOCK_N: tl.constexpr,\n    BLOCK_K: tl.constexpr,\n):\n    pid = tl.program_id(axis=0)\n    grid_m = (M + BLOCK_M - 1) // BLOCK_M\n    grid_n = (N + BLOCK_N - 1) // BLOCK_N\n    pid_m = pid // grid_n\n    pid_n = pid % grid_n\n\n    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    offs_k = tl.arange(0, BLOCK_K)\n\n    # Allocate shared memory for 2-stage pipeline\n    a_shared = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)\n    b_shared = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)\n    a_shared_next = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)\n    b_shared_next = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)\n\n    # Initialize pointers\n    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)\n    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)\n\n    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n\n    # Prologue: Load first stage\n    k_idx = 0 + offs_k\n    a_mask = k_idx[None, :] < K\n    b_mask = k_idx[:, None] < K\n    a_shared = tl.load(a_ptrs, mask=a_mask, other=0.0)\n    b_shared = tl.load(b_ptrs, mask=b_mask, other=0.0)\n\n    for k in range(0, K, BLOCK_K):\n        k_idx_next = k + BLOCK_K + offs_k\n        a_mask_next = k_idx_next[None, :] < K\n        b_mask_next = k_idx_next[:, None] < K\n        \n        # Compute current stage\n        accumulator += tl.dot(a_shared, b_shared)\n        \n        # Prefetch next stage into registers (pipelining)\n        if k < K - BLOCK_K:\n            a_shared_next = tl.load(a_ptrs + BLOCK_K * stride_ak, mask=a_mask_next, other=0.0)\n            b_shared_next = tl.load(b_ptrs + BLOCK_K * stride_bk, mask=b_mask_next, other=0.0)\n        \n        # Move pointers forward\n        a_ptrs += BLOCK_K * stride_ak\n        b_ptrs += BLOCK_K * stride_bk\n        \n        # Swap current and next stage\n        a_shared, a_shared_next = a_shared_next, a_shared\n        b_shared, b_shared_next = b_shared_next, b_shared\n\n    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)\n    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]\n    tl.store(c_ptrs, accumulator.to(tl.float16), mask=mask_c)\n\ndef matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimension mismatch: {K} vs {K2}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c\n    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"\n    M, K = a.shape\n    K2, N = b.shape\n    assert K == K2, f"Inner dimension mismatch: {K} vs {K2}"\n    c = torch.empty((M, N), device=a.device, dtype=a.dtype)\n\n    BLOCK_M = 64\n    BLOCK_N = 64\n    BLOCK_K = 32\n    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)\n\n    matmul_kernel[grid](\n        a, b, c,\n        M, N, K,\n        a.stride(0), a.stride(1),\n        b.stride(0), b.stride(1),\n        c.stride(0), c.stride(1),\n        BLOCK_M=BLOCK_M,\n        BLOCK_N=BLOCK_N,\n        BLOCK_K=BLOCK_K,\n    )\n    return c\n\n# Test for matmul\ndef test_matmul():\n    results = {}\n    M, K, N = 256, 128, 256\n\n    # Test case 1: torch.float16\n    a = torch.randn((M, K), dtype=torch.float16, device='cuda')\n    b = torch.randn((K, N), dtype=torch.float16, device='cuda')\n    c = matmul(a, b)\n    results['test_case_1'] = c\n\n    return results\n\n# Run all tests\nresult_gold = test_matmul()\n
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