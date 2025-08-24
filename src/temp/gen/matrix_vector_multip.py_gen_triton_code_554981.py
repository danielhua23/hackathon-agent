
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_an,
    stride_bn,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        current_n = start_n + offs_n
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + current_n[None, :] * stride_an)
        b_ptrs = B_ptr + current_n * stride_bn
        mask = (offs_m[:, None] < M) & (current_n[None, :] < N)

        a_vals = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
        b_vals = tl.load(b_ptrs, mask=current_n < N, other=0.0).to(tl.float32)

        acc += tl.sum(a_vals * b_vals[None, :], axis=1)

    c_ptrs = C_ptr + offs_m * stride_cm
    mask = offs_m < M
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


def mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2, "A must be 2-D (M x N)"
    assert B.dim() == 1, "B must be 1-D (N)"
    M, N = A.shape
    assert B.shape[0] == N, "B must have same length as N dimension of A"

    C = torch.empty(M, device=A.device, dtype=A.dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

    mv_kernel[grid](
        A, B, C,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0),
        C.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
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
