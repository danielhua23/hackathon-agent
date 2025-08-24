import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=8)], key=['SIZE_M', 'D_HEAD'])
@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < SIZE_M
    mask_n = offs_n < D_HEAD
    mask_in = mask_m[:, None] & mask_n[None, :]
    ptrs_in = M + offs_m[:, None] * matrix_stridex + offs_n[None, :] * matrix_stridey
    tile = tl.load(ptrs_in, mask=mask_in)
    mask_out = mask_n[:, None] & mask_m[None, :]
    ptrs_out = Out + offs_n[:, None] * out_stridex + offs_m[None, :] * out_stridey
    tl.store(ptrs_out, tile, mask=mask_out)

@torch.no_grad()
def wrapper(SIZE_M: int=1024, D_HEAD: int=1024):
    M = torch.randn(SIZE_M, D_HEAD, dtype=torch.float16, device='cpu')
    M = M.to('cuda')
    out = torch.empty(D_HEAD, SIZE_M, dtype=torch.float16, device='cuda')
    grid = lambda META: (triton.cdiv(SIZE_M, META['BLOCK_M']), triton.cdiv(D_HEAD, META['BLOCK_N']))
    kernel[grid](M, out, M.stride(0), M.stride(1), out.stride(0), out.stride(1), SIZE_M, D_HEAD)
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