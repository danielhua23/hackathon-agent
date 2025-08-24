
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
    D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0) * BLOCK_M
    pid_d = tl.program_id(1) * BLOCK_D

    m_offs = pid_m + tl.arange(0, BLOCK_M)
    d_offs = pid_d + tl.arange(0, BLOCK_D)

    mask_m = m_offs < SIZE_M
    mask_d = d_offs < D_HEAD

    src_ptrs = M + d_offs[None, :] * matrix_stridey + m_offs[:, None] * matrix_stridex
    mask = mask_m[:, None] & mask_d[None, :]
    src = tl.load(src_ptrs, mask=mask)

    dst_ptrs = Out + m_offs[None, :] * out_stridex + d_offs[:, None] * out_stridey
    tl.store(dst_ptrs, src, mask=mask.T)

@torch.no_grad()
def wrapper(SIZE_M: int, D_HEAD: int):
    BLOCK_M = 32
    BLOCK_D = 32
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device="cuda")
    out = torch.empty((D_HEAD, SIZE_M), dtype=torch.float16, device="cuda")

    grid = lambda META: (triton.cdiv(SIZE_M, META['BLOCK_M']),
                         triton.cdiv(D_HEAD, META['BLOCK_D']))

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