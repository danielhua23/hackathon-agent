
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey,
    SIZE_M, D_HEAD,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < SIZE_M
    mask_n = offs_n < D_HEAD

    in_ptrs = M + offs_m[:, None] * matrix_stridex + offs_n[None, :] * matrix_stridey
    out_ptrs = Out + offs_n[:, None] * out_stridex + offs_m[None, :] * out_stridey

    load_mask = mask_m[:, None] & mask_n[None, :]
    load_data = tl.load(in_ptrs, mask=load_mask, other=0.0)

    store_mask = mask_n[:, None] & mask_m[None, :]
    tl.store(out_ptrs, load_data.trans(), mask=store_mask)

def wrapper(
    matrix: torch.Tensor, out: torch.Tensor,
    matrix_stridex: int, matrix_stridey: int,
    out_stridex: int, out_stridey: int,
    SIZE_M: int, D_HEAD: int
):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    grid = lambda META: (
        triton.cdiv(SIZE_M, META["BLOCK_SIZE_M"]),
        triton.cdiv(D_HEAD, META["BLOCK_SIZE_N"]),
    )

    kernel[grid](
        matrix, out,
        matrix_stridex, matrix_stridey,
        out_stridex, out_stridey,
        SIZE_M, D_HEAD,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
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