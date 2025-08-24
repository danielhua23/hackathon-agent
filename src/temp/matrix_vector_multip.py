
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(
    A, B, C,
    N, M,
    stride_a0, stride_a1,
    stride_b0,
    stride_c0,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, M, BLOCK_M):
        offs_k = k + offs_m
        mask_a = (offs_n[:, None] < N) & (offs_k[None, :] < M)
        a_ptrs = A + offs_n[:, None] * stride_a0 + offs_k[None, :] * stride_a1
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)

        mask_b = offs_k < M
        b_ptrs = B + offs_k * stride_b0
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)

        accumulator += tl.sum(a * b[None, :], axis=1)

    mask_c = offs_n < N
    c_ptrs = C + offs_n * stride_c0
    tl.store(c_ptrs, accumulator.to(C.type.element_ty), mask=mask_c)


def mv(A: torch.Tensor, B: torch.Tensor):
    assert A.dim() == 2
    assert B.dim() == 1
    N, M = A.shape
    assert B.shape[0] == M

    C = torch.empty((N,), dtype=A.dtype, device=A.device)

    BLOCK_N = 64
    BLOCK_M = 64

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)

    mv_kernel[grid](
        A, B, C,
        N, M,
        A.stride(0), A.stride(1),
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
