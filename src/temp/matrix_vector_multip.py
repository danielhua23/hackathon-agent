\nimport torch\nimport triton\nimport triton.language as tl\n\n\n@triton.autotune(\n    configs=[\n        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}),\n        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}),\n        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}),\n        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}),\n    ],\n    key=['M', 'N']\n)\n@triton.jit\ndef mv_kernel(A_ptr, B_ptr, C_ptr, stride_am, stride_an, stride_b, stride_c, N, M, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):\n    pid_n = tl.program_id(0)\n    pid_m = tl.program_id(1)\n\n    # Start indices\n    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n\n    mask_n = n_offsets < N\n    mask_m = m_offsets < M\n\n    # Compute the starting address\n    a_block_ptr = A_ptr + n_offsets[:, None] * stride_am + m_offsets[None, :] * stride_an\n    b_block_ptr = B_ptr + m_offsets * stride_b\n\n    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)\n\n    for k in range(0, M, BLOCK_M):\n        k_offsets = k + m_offsets\n        mask_km = k_offsets < M\n\n        a = tl.load(a_block_ptr, mask=mask_n[:, None] & mask_km[None, :], other=0.0)\n        b = tl.load(b_block_ptr, mask=mask_km, other=0.0)\n\n        acc += tl.sum(a * b[None, :], axis=1)\n\n        a_block_ptr += BLOCK_M * stride_an\n        b_block_ptr += BLOCK_M * stride_b\n\n    c_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    mask_c = c_offsets < N\n    tl.store(C_ptr + c_offsets * stride_c, acc, mask=mask_c)\n\n\ndef mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.ndim == 2, "A must be 2D"
    assert B.ndim == 1, "B must be 1D"
    assert A.shape[1] == B.shape[0], f"Matrix-vector shape mismatch: {A.shape} vs {B.shape}"

    N, M = A.shape
    C = torch.empty((N,), dtype=A.dtype, device=A.device)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), 1)
    mv_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0),
        C.stride(0),
        N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return C\n    assert A.ndim == 2, "A must be 2D"\n    assert B.ndim == 1, "B must be 1D"\n    assert A.shape[1] == B.shape[0], f"Matrix-vector shape mismatch: {A.shape} vs {B.shape}"\n\n    N, M = A.shape\n    C = torch.empty((N,), dtype=A.dtype, device=A.device)\n\n    BLOCK_M = 64\n    BLOCK_N = 64\n\n    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), 1)\n    mv_kernel[grid](\n        A, B, C,\n        A.stride(0), A.stride(1),\n        B.stride(0),\n        C.stride(0),\n        N, M,\n        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N\n    )\n    return C\n\n\ndef test_mv():\n    # 测试用例 2: 4x3 矩阵与 3x1 向量相乘\n    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device='cuda')\n    B = torch.tensor([1.0, 2.0, 3.0], device='cuda')\n    triton_result_2 = mv(A, B)\n\n    # 测试用例 3: 32x16 矩阵与 16x1 向量相乘\n    A = torch.randn(32, 16, device='cuda')\n    B = torch.randn(16, device='cuda')\n    triton_result_3 = mv(A, B)\n\n    return {\n        "test_case_2": triton_result_2,\n        "test_case_3": triton_result_3,\n    }\n\nresult_gold = test_mv()
##################################################################################################################################################



def test_mv():
    # 测试用例 2: 4x3 矩阵与 3x1 向量相乘
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device='cuda')
    B = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    triton_result_2 = mv(A, B)

    # 测试用例 3: 32x16 矩阵与 16x1 向量相乘
    A = torch.randn(32, 16, device='cuda')
    B = torch.randn(16, device='cuda')
    triton_result_3 = mv(A, B)

    return {
        "test_case_2": triton_result_2,
        "test_case_3": triton_result_3,
    }

result_gold = test_mv()
