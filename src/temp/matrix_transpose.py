
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute base pointers for this block
    offs_m = pid_m * 16 + tl.arange(0, 16)
    offs_n = pid_n * 16 + tl.arange(0, 16)

    # Mask to prevent out-of-bounds access
    mask = (offs_m[:, None] < SIZE_M) & (offs_n[None, :] < D_HEAD)

    # Compute memory addresses
    in_ptrs = M + offs_m[:, None] * matrix_stridex + offs_n[None, :] * matrix_stridey
    out_ptrs = Out + offs_n[:, None] * out_stridex + offs_m[None, :] * out_stridey

    # Load and transpose
    data = tl.load(in_ptrs, mask=mask)
    tl.store(out_ptrs, data, mask=mask)


def wrapper(matrix_stridex: int, matrix_stridey: int, out_stridex: int, out_stridey: int):
    # Set dimensions
    SIZE_M = 512
    D_HEAD = 256
    
    # Initialize tensors on device
    matrix = torch.randn(SIZE_M, D_HEAD, dtype=torch.float16, device='cuda')
    out = torch.zeros(D_HEAD, SIZE_M, dtype=torch.float16, device='cuda')
    
    # Configure grid
    grid = lambda META: (
        triton.cdiv(SIZE_M, 16),
        triton.cdiv(D_HEAD, 16)
    )
    
    # Launch kernel
    kernel[grid](
        matrix, out,
        matrix.stride(0), matrix.stride(1),
        out.stride(0), out.stride(1),
        SIZE_M, D_HEAD
    )
    
    return out

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