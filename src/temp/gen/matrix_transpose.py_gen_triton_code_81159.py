
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
    ROW_TILE: tl.constexpr,
    COL_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * ROW_TILE + tl.arange(0, ROW_TILE)
    offs_n = pid_n * COL_TILE + tl.arange(0, COL_TILE)

    mask_m = offs_m < SIZE_M
    mask_n = offs_n < D_HEAD
    mask = mask_m[:, None] & mask_n[None, :]

    rows = offs_m[:, None]
    cols = offs_n[None, :]

    m_addrs = M + rows * matrix_stridex + cols * matrix_stridey
    x = tl.load(m_addrs, mask=mask)

    out_addrs = Out + cols * out_stridex + rows * out_stridey
    tl.store(out_addrs, x, mask=mask.T)


def wrapper(SIZE_M: int, D_HEAD: int):
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    ROW_TILE = 16
    COL_TILE = 16
    grid = lambda META: (
        triton.cdiv(SIZE_M, META['ROW_TILE']),
        triton.cdiv(D_HEAD, META['COL_TILE']),
    )

    kernel[grid](
        matrix,
        out,
        matrix.stride(0),
        matrix.stride(1),
        out.stride(0),
        out.stride(1),
        SIZE_M=SIZE_M,
        D_HEAD=D_HEAD,
        ROW_TILE=ROW_TILE,
        COL_TILE=COL_TILE,
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