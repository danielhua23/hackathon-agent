
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(A, B, C, stride_am, stride_an, stride_b, stride_cm,
              N: tl.constexpr, M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_N
    offs_n = block_start + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    mask_n = offs_n < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for start_m in range(0, M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        mask_m = offs_m_curr < M
        a_ptrs = A + (offs_n[:, None] * stride_am + offs_m_curr[None, :] * stride_an)
        b_ptrs = B + offs_m_curr * stride_b
        a = tl.load(a_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_m, other=0.0)
        acc += tl.sum(a * b[None, :], axis=1)

    c_ptrs = C + offs_n * stride_cm
    tl.store(c_ptrs, acc, mask=mask_n)


def mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2, "A must be 2D"
    assert B.dim() == 1, "B must be 1D"
    N, M = A.shape
    assert B.shape[0] == M, "A and B shapes incompatible"

    C = torch.empty(N, dtype=A.dtype, device=A.device)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)

    mv_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1), B.stride(0), C.stride(0),
        N, M,
        BLOCK_N=64, BLOCK_M=64,
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
