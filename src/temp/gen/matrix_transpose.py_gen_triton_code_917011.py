
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    mask = (offs_m < SIZE_M) & (offs_n < D_HEAD)

    m_ptrs = M + offs_m * matrix_stridex + offs_n * matrix_stridey
    out_ptrs = Out + offs_n * out_stridex + offs_m * out_stridey

    m_val = tl.load(m_ptrs, mask=mask)
    tl.store(out_ptrs, m_val, mask=mask)

def wrapper(size_m: int, d_head: int):
    SIZE_M = size_m
    D_HEAD = d_head
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    BLOCK_M = 16
    BLOCK_N = 16
    grid = (triton.cdiv(SIZE_M, BLOCK_M), triton.cdiv(D_HEAD, BLOCK_N))
    kernel[grid](
        matrix, out,
        matrix.stride(0), matrix.stride(1),
        out.stride(0), out.stride(1),
        SIZE_M, D_HEAD,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
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