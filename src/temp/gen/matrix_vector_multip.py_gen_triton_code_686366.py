
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(A, B, C, M, N, stride_am, stride_an, stride_b, stride_c,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Compute block pointer for C
    c_ptrs = C + offs_m * stride_c
    mask_m = offs_m < M
    c_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load A (BLOCK_M x BLOCK_N)
    a_blk_ptrs = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    # Load and compute
    for start_n in range(0, N, BLOCK_N):
        # Offset to current block in N
        curr_n = start_n + offs_n
        mask_n = curr_n < N
        a_ptrs = a_blk_ptrs + start_n * stride_an
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)

        b_ptrs = B + curr_n * stride_b
        b = tl.load(b_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        c_acc += tl.sum(a.to(tl.float32) * b[None, :], axis=1)

    tl.store(c_ptrs, c_acc.to(C.type.element_ty), mask=mask_m)


def mv(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor = None):
    assert A.dim() == 2 and B.dim() == 1, "A must be 2-D and B must be 1-D"
    M, N = A.shape
    assert B.shape[0] == N, "Inner dimensions must match"
    if C is None:
        C = torch.empty(M, dtype=A.dtype, device=A.device)

    stride_am = A.stride(0)
    stride_an = A.stride(1)
    stride_b = B.stride(0)
    stride_c = C.stride(0)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    mv_kernel[grid](
        A, B, C, M, N,
        stride_am, stride_an, stride_b, stride_c,
        BLOCK_M=64, BLOCK_N=32
    )
    return C


##################################################################################################################################################





def test_mv():

    # 测试用例 2: 4x3 矩阵与 3x1 向量相乘

    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device='cuda')

    B = torch.tensor([1.0, 2.0, 3.0], device='cuda')

    triton_result_2 = mv(A, B)



    # 测试用例 3: 32x16 矩阵与 16x1 向量相乘

    A = torch.randn(32, 16, device='cuda')

    B = torch.randn(16, device='cuda')

    triton_result_3 = mv(A, B)



    return {

        "test_case_2": triton_result_2,

        "test_case_3": triton_result_3,

    }



result_gold = test_mv()
