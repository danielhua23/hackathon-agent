
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD):
    BLOCK_SIZE = 32
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    rn = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]

    mask = (rm < SIZE_M) & (rn < D_HEAD)
    in_idx = rm * matrix_stridex + rn * matrix_stridey
    out_idx = rn * out_stridex + rm * out_stridey

    val = tl.load(M + in_idx, mask=mask)
    tl.store(Out + out_idx, val, mask=mask)


def wrapper(SIZE_M: int, D_HEAD: int) -> torch.Tensor:
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device="cuda")
    out = torch.empty((D_HEAD, SIZE_M), dtype=torch.float16, device="cuda")

    grid = lambda META: (triton.cdiv(SIZE_M, 32), triton.cdiv(D_HEAD, 32))

    kernel[grid](
        matrix,
        out,
        matrix.stride(0),
        matrix.stride(1),
        out.stride(0),
        out.stride(1),
        SIZE_M,
        D_HEAD
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