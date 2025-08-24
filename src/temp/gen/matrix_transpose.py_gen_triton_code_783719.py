
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
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = offs_m < SIZE_M
    mask_d = offs_d < D_HEAD
    mask = mask_m[:, None] & mask_d[None, :]

    m_ptrs = M + offs_m[:, None] * matrix_stridey + offs_d[None, :] * matrix_stridex
    o_ptrs = Out + offs_d[:, None] * out_stridex + offs_m[None, :] * out_stridey

    vals = tl.load(m_ptrs, mask=mask)
    tl.store(o_ptrs, vals.T, mask=mask.T)

def wrapper(SIZE_M, D_HEAD):
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device="cuda")
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device="cuda")

    BLOCK_M = 32
    BLOCK_D = 32
    grid = (
        triton.cdiv(SIZE_M, BLOCK_M),
        triton.cdiv(D_HEAD, BLOCK_D),
    )

    kernel[grid](
        matrix,
        out,
        matrix.stride(1),
        matrix.stride(0),
        out.stride(1),
        out.stride(0),
        SIZE_M,
        D_HEAD,
        BLOCK_M,
        BLOCK_D,
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