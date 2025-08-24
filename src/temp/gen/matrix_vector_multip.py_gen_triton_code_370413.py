
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(
    A, B, C,
    N, M,
    stride_am, stride_an,
    stride_b,
    stride_c,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_N
    offs_n = block_start + tl.arange(0, BLOCK_N)
    col_mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for mid in range(0, M, BLOCK_M):
        offs_m = mid + tl.arange(0, BLOCK_M).to(tl.int32)

        # Compute A pointers [BLOCK_N, BLOCK_M]
        a_ptrs = A + offs_n[:, None] * stride_am + offs_m[None, :] * stride_an
        mask_a = col_mask[:, None] & (offs_m[None, :] < M)

        a_block = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Compute B pointers [BLOCK_M]
        b_ptrs = B + offs_m * stride_b
        mask_b = offs_m < M
        b_block = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Reduce along block_m dimension
        acc += tl.sum(a_block * b_block[None, :], axis=1).to(tl.float32)

    # Store
    c_ptrs = C + offs_n * stride_c
    tl.store(c_ptrs, acc.to(C.type.element_ty), mask=col_mask)


def mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2, "A must be a 2D matrix (N x M)"
    assert B.dim() == 1, "B must be a 1D vector (M)"
    assert A.shape[1] == B.shape[0], "Matrix-vector dimension mismatch"

    N, M = A.shape
    C = torch.empty((N,), dtype=A.dtype, device=A.device)

    BLOCK_N = 64
    BLOCK_M = 64

    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), )

    mv_kernel[grid](
        A, B, C,
        N, M,
        A.stride(0), A.stride(1),
        B.stride(0),
        C.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M
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
