
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
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * D_HEAD + tl.arange(0, D_HEAD)
    offs_d = pid_d * SIZE_M + tl.arange(0, SIZE_M)

    mask_m = offs_m < SIZE_M * D_HEAD
    mask_d = offs_d < SIZE_M * D_HEAD

    src_ptrs = M + offs_m[:, None] * matrix_stridex + offs_d[None, :] * matrix_stridey
    src_mask = (offs_m[:, None] < SIZE_M) & (offs_d[None, :] < D_HEAD)

    src_data = tl.load(src_ptrs, mask=src_mask, other=0.0)

    dst_ptrs = Out + offs_d[:, None] * out_stridex + offs_m[None, :] * out_stridey
    dst_mask = (offs_d[:, None] < D_HEAD) & (offs_m[None, :] < SIZE_M)

    tl.store(dst_ptrs, src_data.T, mask=dst_mask)

@torch.no_grad()
def wrapper(SIZE_M: int, D_HEAD: int):
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device="cuda")
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device="cuda")

    grid = (1,)
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