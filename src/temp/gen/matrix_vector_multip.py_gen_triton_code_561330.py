
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(A, B, C, M, N, stride_am, stride_an, stride_b, stride_c, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(0, N, BLOCK_N):
        rn_k = k + rn
        mask_a = (rm[:, None] < M) & (rn_k[None, :] < N)
        mask_b = rn_k < N

        a_ptrs = A + (rm[:, None] * stride_am + rn_k[None, :] * stride_an)
        b_ptrs = B + rn_k * stride_b

        a_block = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)
        b_block = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)

        acc += tl.sum(a_block * b_block[None, :], axis=1)

    mask_c = rm < M
    c_ptrs = C + rm * stride_c
    tl.store(c_ptrs, acc, mask=mask_c)


def mv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2
    assert b.dim() == 1
    assert a.size(1) == b.size(0)

    M, N = a.shape
    C = torch.empty(M, dtype=a.dtype, device=a.device)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

    mv_kernel[grid](
        a, b, C,
        M, N,
        a.stride(0), a.stride(1),
        b.stride(0),
        C.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
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
