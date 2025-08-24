import torch
import triton
import triton.language as tl

@triton.jit
def kernel(M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey, SIZE_M: tl.constexpr, D_HEAD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < SIZE_M
    mask_d = offs_d < D_HEAD
    matrix_ptrs = M + (offs_m[:, None] * matrix_stridey + offs_d[None, :] * matrix_stridex)
    out_ptrs = Out + (offs_m[None, :] * out_stridex + offs_d[:, None] * out_stridey)
    matrix_tile = tl.load(matrix_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    tl.store(out_ptrs, matrix_tile.T, mask=mask_m[None, :] & mask_d[:, None])

def wrapper(SIZE_M: int=1024, D_HEAD: int=64):
    device = torch.device('cuda')
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device=device)
    out = torch.empty((D_HEAD, SIZE_M), dtype=torch.float16, device=device)
    BLOCK_M = 32
    BLOCK_D = 32
    grid = lambda META: (triton.cdiv(SIZE_M, META['BLOCK_M']), triton.cdiv(D_HEAD, META['BLOCK_D']))
    kernel[grid](matrix, out, matrix.stride(0), matrix.stride(1), out.stride(0), out.stride(1), SIZE_M, D_HEAD, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D)
    return out
if __name__ == '__main__':
    M = 4
    D = 6
    inp = torch.randn((M, D), dtype=torch.float16, device='cuda')
    reference = inp.t()
    kernel_result = wrapper(M, D)
    assert torch.allclose(reference, kernel_result)

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