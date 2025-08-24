
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    grid_m = tl.cdiv(SIZE_M, 16)
    grid_n = tl.cdiv(D_HEAD, 16)

    for i in range(0, grid_m):
        for j in range(0, grid_n):
            offs_m = i * 16 + tl.arange(0, 16)[:, None]
            offs_n = j * 16 + tl.arange(0, 16)[None, :]

            mask = (offs_m < SIZE_M) & (offs_n < D_HEAD)

            m_ptrs = M + pid_m * matrix_stridex + pid_n * matrix_stridey + offs_m * matrix_stridex + offs_n * matrix_stridey
            out_ptrs = Out + pid_m * out_stridex + pid_n * out_stridey + offs_n * out_stridex + offs_m * out_stridey

            m_val = tl.load(m_ptrs, mask=mask, other=0.0)
            tl.store(out_ptrs, m_val, mask=mask)

def wrapper():
    SIZE_M = 64
    D_HEAD = 128
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    grid = (1,)
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