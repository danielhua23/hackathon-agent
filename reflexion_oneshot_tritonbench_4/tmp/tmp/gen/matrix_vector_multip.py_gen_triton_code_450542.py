import torch
import triton
import triton.language as tl

@triton.jit
def mv_kernel(A, B, C, M, N, stride_am, stride_an, stride_b, stride_c, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_a = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        mask_b = offs_n < N
        a_blk = tl.load(A + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask_a, other=0.0)
        b_vec = tl.load(B + offs_n * stride_b, mask=mask_b, other=0.0)
        product = a_blk * b_vec[None, :]
        acc += tl.sum(product, axis=1)
    offs_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_out = offs_out < M
    tl.store(C + offs_out * stride_c, acc.to(C.dtype.element_ty), mask=mask_out)

def mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 1
    M, N = A.shape
    assert B.size(0) == N
    C = torch.empty(M, dtype=A.dtype, device=A.device)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    mv_kernel[grid](A, B, C, M, N, A.stride(0), A.stride(1), B.stride(0), C.stride(0), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
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
