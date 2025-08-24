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
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = offs_m < SIZE_M
    mask_n = offs_n < D_HEAD

    idx = (offs_m[:, None] * matrix_stridey + offs_n[None, :] * matrix_stridex)
    matrix_data = tl.load(M + idx, mask=mask_m[:, None] & mask_n[None, :])

    idx_out = (offs_n[:, None] * out_stridey + offs_m[None, :] * out_stridex)
    tl.store(Out + idx_out, matrix_data, mask=mask_n[:, None] & mask_m[None, :])

def wrapper(SIZE_M: int, D_HEAD: int):
    matrix = torch.randn(SIZE_M, D_HEAD, dtype=torch.float16, device='cuda')
    out = torch.zeros(D_HEAD, SIZE_M, dtype=torch.float16, device='cuda')

    BLOCK_M = 128
    BLOCK_D = 128

    grid = (triton.cdiv(SIZE_M, BLOCK_M), triton.cdiv(D_HEAD, BLOCK_D))

    kernel[grid](
        matrix,
        out,
        matrix.stride(0),
        matrix.stride(1),
        out.stride(0),
        out.stride(1),
        SIZE_M,
        D_HEAD,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
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