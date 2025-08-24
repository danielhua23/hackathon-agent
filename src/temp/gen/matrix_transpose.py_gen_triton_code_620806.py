
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    BLOCK_SIZE_M = tl.program_id(2) if hasattr(tl.program_id, '__call__') else 32
    BLOCK_SIZE_N = tl.program_id(3) if hasattr(tl.program_id, '__call__') else 32

    offs_m = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    mask_m = offs_m < D_HEAD
    mask_n = offs_n < SIZE_M

    ptrs_m = M + offs_n[:, None] * matrix_stridex + offs_m[None, :] * matrix_stridey
    ptrs_out = Out + offs_m[:, None] * out_stridex + offs_n[None, :] * out_stridey

    x = tl.load(ptrs_m, mask=mask_n[:, None] & mask_m[None, :])
    tl.store(ptrs_out, x, mask=mask_m[:, None] & mask_n[None, :])

def wrapper(SIZE_M: int, D_HEAD: int):
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = lambda META: (
        triton.cdiv(D_HEAD, BLOCK_SIZE_N),
        triton.cdiv(SIZE_M, BLOCK_SIZE_M)
    )

    kernel[grid](
        matrix, out,
        matrix.stride(0), matrix.stride(1),
        out.stride(0), out.stride(1),
        SIZE_M, D_HEAD,
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