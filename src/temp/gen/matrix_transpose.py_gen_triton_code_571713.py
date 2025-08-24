
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
    row = tl.arange(0, SIZE_M)[:, None]
    col = tl.arange(0, D_HEAD)[None, :]
    in_ptrs = M + row * matrix_stridex + col * matrix_stridey
    out_ptrs = Out + col * out_stridex + row * out_stridey
    vals = tl.load(in_ptrs)
    tl.store(out_ptrs, vals)


def wrapper(SIZE_M: int, D_HEAD: int):
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')
    grid = lambda meta: (1,)
    kernel[grid](
        matrix,
        out,
        matrix.stride(0),
        matrix.stride(1),
        out.stride(0),
        out.stride(1),
        SIZE_M,
        D_HEAD,
    )
    return out


if __name__ == "__main__":
    SIZE_M = 64
    D_HEAD = 128
    out = wrapper(SIZE_M, D_HEAD)
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    assert torch.allclose(matrix.t(), out)


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