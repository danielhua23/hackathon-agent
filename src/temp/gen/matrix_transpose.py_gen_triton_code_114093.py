
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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_rows = SIZE_M
    n_cols = D_HEAD

    num_tasks = n_rows * n_cols
    for i in range(pid, num_tasks, BLOCK_SIZE):
        if i < num_tasks:
            row = i // n_cols
            col = i % n_cols

            in_ptr  = M  + tl.make_block_ptr(
                base=M,
                shape=(n_rows, n_cols),
                strides=(matrix_stridex, matrix_stridey),
                offsets=(row, col),
                block_shape=(1, 1),
                order=(0, 1)
            )
            val = tl.load(in_ptr)

            out_ptr = Out + tl.make_block_ptr(
                base=Out,
                shape=(n_cols, n_rows),
                strides=(out_stridex, out_stridey),
                offsets=(col, row),
                block_shape=(1, 1),
                order=(0, 1)
            )
            tl.store(out_ptr, val)


def wrapper(SIZE_M: int, D_HEAD: int):
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device='cuda')
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device='cuda')

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(SIZE_M * D_HEAD, BLOCK_SIZE),)
    kernel[grid](
        matrix,
        out,
        matrix.stride(0),
        matrix.stride(1),
        out.stride(0),
        out.stride(1),
        SIZE_M,
        D_HEAD,
        BLOCK_SIZE,
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