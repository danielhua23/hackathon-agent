
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
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    matrix_ptr = M + d_head_arange[None, :] * matrix_stridex + size_m_arange[:, None] * matrix_stridey
    out_ptr = Out + size_m_arange[None, :] * out_stridex + d_head_arange[:, None] * out_stridey
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, matrix)

def wrapper(SIZE_M: int, D_HEAD: int):
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device="cuda")
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device="cuda")
    grid = (1,)
    kernel[grid](
        matrix,
        out,
        matrix.stride(1),
        matrix.stride(0),
        out.stride(1),
        out.stride(0),
        SIZE_M,
        D_HEAD,
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