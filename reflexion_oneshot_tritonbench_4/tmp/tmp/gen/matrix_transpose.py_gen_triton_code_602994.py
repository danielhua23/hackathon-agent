import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M: tl.constexpr, D_HEAD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < SIZE_M
    mask_n = offs_n < D_HEAD
    mask = mask_m[:, None] & mask_n[None, :]
    src_ptrs = M + offs_m[:, None] * matrix_stridey + offs_n[None, :] * matrix_stridex
    out_ptrs = Out + offs_n[:, None] * out_stridex + offs_m[None, :] * out_stridey
    original_data = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, tl.trans(original_data), mask=tl.trans(mask))

def wrapper(SIZE_M: int, D_HEAD: int) -> torch.Tensor:
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(SIZE_M, BLOCK_M), triton.cdiv(D_HEAD, BLOCK_N))
    kernel[grid](matrix, out, matrix.stride(1), matrix.stride(0), out.stride(1), out.stride(0), SIZE_M=SIZE_M, D_HEAD=D_HEAD, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
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