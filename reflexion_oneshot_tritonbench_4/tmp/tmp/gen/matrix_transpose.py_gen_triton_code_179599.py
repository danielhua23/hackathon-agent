import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M: tl.constexpr, D_HEAD: tl.constexpr):
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    rows = size_m_arange[:, None]
    cols = d_head_arange[None, :]
    matrix_ptr = M + rows * matrix_stridex + cols * matrix_stridey
    matrix_value = tl.load(matrix_ptr)
    out_rows = d_head_arange[:, None]
    out_cols = size_m_arange[None, :]
    out_ptr = Out + out_rows * out_stridex + out_cols * out_stridey
    tl.store(out_ptr, tl.trans(matrix_value))

def wrapper(SIZE_M: int, D_HEAD: int):
    device = 'cuda'
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device=device)
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device=device)
    grid = (1,)
    kernel[grid](matrix, out, matrix.stride(1), matrix.stride(0), out.stride(1), out.stride(0), SIZE_M=SIZE_M, D_HEAD=D_HEAD)
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