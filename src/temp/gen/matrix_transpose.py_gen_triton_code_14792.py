
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
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < SIZE_M
    mask_n = offs_n < D_HEAD

    idxs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    idxs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    matrix_ptrs = M + (idxs_m[:, None] * matrix_stridex + idxs_n[None, :] * matrix_stridey)
    out_ptrs = Out + (idxs_n[:, None] * out_stridex + idxs_m[None, :] * out_stridey)

    mask = mask_m[:, None] & mask_n[None, :]
    a = tl.load(matrix_ptrs, mask=mask)
    tl.store(out_ptrs, a, mask=mask)

def wrapper(SIZE_M: int, D_HEAD: int):
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    BLOCK_M = 32
    BLOCK_N = 32
    grid = lambda meta: (triton.cdiv(SIZE_M, meta['BLOCK_M']),
                         triton.cdiv(D_HEAD, meta['BLOCK_N']))
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
        BLOCK_N=BLOCK_N,
    )
    return out

if __name__ == "__main__":
    SIZE_M = 64
    D_HEAD = 128
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = wrapper(SIZE_M, D_HEAD)
    ref = matrix.t()
    assert torch.allclose(ref, out, atol=1e-3), abs(ref - out).max()


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