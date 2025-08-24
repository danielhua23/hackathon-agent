
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M,
    Out,
    matrix_stridex,
    matrix_stridey,
    out_stridex,
    out_stridey,
    SIZE_M,
    D_HEAD
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * 128 + tl.arange(0, 128)
    offs_n = pid_n * 128 + tl.arange(0, 128)
    
    mask_m = offs_m < SIZE_M
    mask_n = offs_n < D_HEAD
    
    # Load data
    idx = offs_m[:, None] * matrix_stridey + offs_n[None, :] * matrix_stridex
    data = tl.load(M + idx, mask=mask_m[:, None] & mask_n[None, :])
    
    # Store transposed
    idx_out = offs_n[:, None] * out_stridey + offs_m[None, :] * out_stridex
    tl.store(Out + idx_out, data, mask=mask_n[:, None] & mask_m[None, :])

def wrapper():
    SIZE_M = 1024
    D_HEAD = 512
    
    # ROCm uses "cuda" device name
    matrix = torch.randn(SIZE_M, D_HEAD, dtype=torch.float16, device='cuda')
    out = torch.zeros(D_HEAD, SIZE_M, dtype=torch.float16, device='cuda')
    
    grid = (triton.cdiv(SIZE_M, 128), triton.cdiv(D_HEAD, 128))
    
    kernel[grid](
        matrix, out,
        matrix.stride(1), matrix.stride(0),
        out.stride(1), out.stride(0),
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