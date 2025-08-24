\n\nimport torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey,\n           SIZE_M, D_HEAD, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):\n    pid_row = tl.program_id(0)\n    pid_col = tl.program_id(1)\n\n    rows = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)\n    cols = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)\n\n    mask_rows = rows < SIZE_M\n    mask_cols = cols < D_HEAD\n\n    offset_m = rows[:, None] * matrix_stridey + cols[None, :] * matrix_stridex\n    offset_out = rows[:, None] * out_stridex + cols[None, :] * out_stridey\n\n    mask = mask_rows[:, None] & mask_cols[None, :]\n\n    x = tl.load(M + offset_m, mask=mask)\n\n    smem = tl.alloc_block(BLOCK_M * BLOCK_N, dtype=x.dtype, scope="shared")\n    smem_idx = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]\n    tl.store(smem + smem_idx, x, mask=mask)\n    tl.debug_barrier()\n\n    x_t = tl.load(smem + smem_idx)\n    tl.store(Out + offset_out, x_t, mask=mask)\n\ndef wrapper(size_m, d_head):
    dtype = torch.float16
    matrix = torch.randn((size_m, d_head), dtype=dtype, device='hip')
    out = torch.empty((size_m, d_head), dtype=dtype, device='hip')

    BLOCK_M = 16
    BLOCK_N = 16
    grid = lambda META: (triton.cdiv(size_m, META['BLOCK_M']),
                         triton.cdiv(d_head, META['BLOCK_N']))

    kernel[grid](
        matrix, out,
        matrix.stride(0), matrix.stride(1),
        out.stride(0), out.stride(1),
        size_m, d_head,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4
    )

    return out.t().contiguous().t()\n    dtype = torch.float16\n    matrix = torch.randn((size_m, d_head), dtype=dtype, device='hip')\n    out = torch.empty((size_m, d_head), dtype=dtype, device='hip')\n\n    BLOCK_M = 16\n    BLOCK_N = 16\n    grid = lambda META: (triton.cdiv(size_m, META['BLOCK_M']),\n                         triton.cdiv(d_head, META['BLOCK_N']))\n\n    kernel[grid](\n        matrix, out,\n        matrix.stride(0), matrix.stride(1),\n        out.stride(0), out.stride(1),\n        size_m, d_head,\n        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,\n        num_warps=4\n    )\n\n    return out.t().contiguous().t()\n\ndef test_triton_vs_torch():\n    results = {}\n\n    # 测试用例 1: 基本矩阵转置 (小矩阵)\n    size_m, d_head = 16, 16\n    out = wrapper(size_m, d_head)\n    results["test_case_1"] = out.clone()\n\n    # 测试用例 2: 非方形矩阵\n    size_m, d_head = 32, 64\n    out = wrapper(size_m, d_head)\n    results["test_case_2"] = out.clone()\n\n    return results\n\n# 运行测试\nresult_gold = test_triton_vs_torch()\n# print(result_gold)\n
##################################################################################################################################################



import torch

def test_triton_vs_torch():
    results = {}

    # 测试用例 1: 基本矩阵转置 (小矩阵)
    size_m, d_head = 16, 16
    out = wrapper(size_m, d_head)
    results["test_case_1"] = out.clone()

    # 测试用例 2: 非方形矩阵
    size_m, d_head = 32, 64
    out = wrapper(size_m, d_head)
    results["test_case_2"] = out.clone()

    return results


# 运行测试
result_gold = test_triton_vs_torch()
# print(result_gold)