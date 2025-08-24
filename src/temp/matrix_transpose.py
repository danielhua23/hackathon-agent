
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD):
    pid = tl.program_id(0)
    num_blocks_h = (D_HEAD + 127) // 128
    block_m = pid // num_blocks_h
    block_d = pid % num_blocks_h

    BLOCK_M = 64
    BLOCK_D = 128

    rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = block_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = rm < SIZE_M
    mask_d = rd < D_HEAD
    mask = mask_m[:, None] & mask_d[None, :]

    ptrs_in = M + rm[:, None] * matrix_stridex + rd[None, :] * matrix_stridey
    vals = tl.load(ptrs_in, mask=mask)
    ptrs_out = Out + rd[None, :] * out_stridex + rm[:, None] * out_stridey
    tl.store(ptrs_out, vals, mask=mask)

def wrapper(size_m: int, d_head: int):
    torch.manual_seed(0)
    M = torch.randn(size_m, d_head, dtype=torch.float16, device='cuda')
    Out = torch.zeros(d_head, size_m, dtype=torch.float16, device='cuda')

    grid = lambda meta: [((size_m + 63) // 64) * ((d_head + 127) // 128)]

    kernel[grid](
        M,
        Out,
        M.stride(0),
        M.stride(1),
        Out.stride(0),
        Out.stride(1),
        size_m,
        d_head,
    )
    return Out

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