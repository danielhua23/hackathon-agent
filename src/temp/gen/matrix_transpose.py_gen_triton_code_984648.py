
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out,
           matrix_stridex, matrix_stridey,
           out_stridex, out_stridey,
           SIZE_M: tl.constexpr, D_HEAD: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    cols = tl.arange(0, SIZE_M)
    rows = tl.arange(0, D_HEAD)

    mask = (cols[:, None] < SIZE_M) & (rows[None, :] < D_HEAD)

    m_ptrs = M + cols[:, None] * matrix_stridey + rows[None, :] * matrix_stridex
    o_ptrs = Out + rows[None, :] * out_stridey + cols[:, None] * out_stridex
    
    vals = tl.load(m_ptrs, mask=mask)
    tl.store(o_ptrs, vals, mask=mask)

def wrapper():
    SIZE_M = 128
    D_HEAD = 64
    
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    grid = (1,)
    kernel[grid](
        matrix, out,
        matrix.stride(1), matrix.stride(0),
        out.stride(1), out.stride(0),
        SIZE_M, D_HEAD,
        num_warps=4,
        num_stages=2
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