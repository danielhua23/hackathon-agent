
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M, Out, matrix_stridex, matrix_stridey, out_stridex, out_stridey,
    SIZE_M, D_HEAD,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    x_idx = offs_m[:, None] * D_HEAD + offs_n[None, :]
    y_idx = offs_n[:, None] * SIZE_M + offs_m[None, :]

    mask_i = (offs_m[:, None] < SIZE_M) & (offs_n[None, :] < D_HEAD)
    mask_o = (offs_n[:, None] < D_HEAD) & (offs_m[None, :] < SIZE_M)

    val = tl.load(M + x_idx, mask=mask_i, other=0.0)
    tl.store(Out + y_idx, val.trans(), mask=mask_o)

def wrapper(
    SIZE_M: int, D_HEAD: int
) -> torch.Tensor:
    device = torch.device("cuda")
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device=device)
    out = torch.empty((D_HEAD, SIZE_M), dtype=torch.float16, device=device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (
        triton.cdiv(SIZE_M, BLOCK_SIZE_M),
        triton.cdiv(D_HEAD, BLOCK_SIZE_N),
    )

    # For row-major, matrix.stride(1) = D_HEAD, matrix.stride(0) = 1  => stride_x = D_HEAD, stride_y = 1
    #            out.stride(1) = SIZE_M,  out.stride(0) = 1      => stride_x_out = SIZE_M, stride_y_out = 1
    # tl.arange * stride simplifies when stride==1 and we use element offsets; so we pass stride=1.
    kernel[grid](
        matrix, out,
        1, 1,
        1, 1,
        SIZE_M, D_HEAD,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
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