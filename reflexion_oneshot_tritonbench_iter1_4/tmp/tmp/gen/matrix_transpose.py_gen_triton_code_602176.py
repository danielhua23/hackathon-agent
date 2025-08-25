import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M, D_HEAD, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < SIZE_M
    mask_d = offs_d < D_HEAD
    inp_ptrs = M + offs_m[:, None] * matrix_stridex + offs_d[None, :] * matrix_stridey
    out_ptrs = Out + offs_d[:, None] * out_stridex + offs_m[None, :] * out_stridey
    tile = tl.load(inp_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    tl.store(out_ptrs, tile.T, mask=mask_d[:, None] & mask_m[None, :])

def wrapper(size_m: int, d_head: int) -> torch.Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrix = torch.randn((size_m, d_head), dtype=torch.float16, device=device)
    out = torch.empty((d_head, size_m), dtype=torch.float16, device=device)
    BLOCK_M = 16
    BLOCK_D = 16
    grid = (triton.cdiv(size_m, BLOCK_M), triton.cdiv(d_head, BLOCK_D))
    kernel[grid](matrix, out, matrix.stride(0), matrix.stride(1), out.stride(0), out.stride(1), size_m, d_head, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D)
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