
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(A, B, C, M, N, stride_am, stride_an, stride_b, stride_c,
              BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    offs_b_base = tl.arange(0, BLOCK_M)

    for k in range(0, M, BLOCK_M):
        offs_k = k + offs_b_base
        mask_A = (offs_m[:, None] < N) & (offs_k[None, :] < M)
        offs_A = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_an
        a_tile = tl.load(offs_A, mask=mask_A, other=0.0)
        mask_B = offs_k < M
        offs_B = B + offs_k * stride_b
        b_vec = tl.load(offs_B, mask=mask_B, other=0.0)
        acc += tl.sum(a_tile * b_vec[None, :], axis=1)

    offs_c = C + offs_m * stride_c
    mask_c = offs_m < N
    tl.store(offs_c, acc.to(C.type.element_ty), mask=mask_c)


def mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2, "Input tensor A must be 2D (N x M)"
    assert B.dim() == 1, "Input tensor B must be 1D"
    N, M = A.shape
    assert B.shape[0] == M, "Incompatible dimensions for MV multiplication"

    C = torch.empty((N,), dtype=A.dtype, device=A.device)

    BLOCK_N = 64
    BLOCK_M = 32

    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), )

    mv_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        C.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
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
