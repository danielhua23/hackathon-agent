
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
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    m_ptrs = M + pid_x * matrix_stridey + pid_y * matrix_stridex
    o_ptrs = Out + pid_y * out_stridey + pid_x * out_stridex
    
    val = tl.load(m_ptrs)
    tl.store(o_ptrs, val)

def wrapper():
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')
    
    grid = lambda meta: (SIZE_M, D_HEAD)
    kernel[grid](
        matrix, out,
        matrix.stride(0), matrix.stride(1),
        out.stride(0), out.stride(1),
        SIZE_M, D_HEAD
    )
    return out

SIZE_M = 64
D_HEAD = 128
result = wrapper()


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