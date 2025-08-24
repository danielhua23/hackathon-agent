import torch
import triton
import triton.language as tl

@triton.jit
def mv_kernel(A, B, C, M, K, stride_am, stride_ak, stride_bk, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(K):
        a = tl.load(A + offs_m * stride_am + k * stride_ak, mask=mask_m, other=0.0)
        b = tl.load(B + k * stride_bk)
        acc += a.to(tl.float32) * b.to(tl.float32)
    tl.store(C + offs_m, acc.to(C.dtype.element_ty), mask=mask_m)

def mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert B.ndim == 1
    M, K = A.shape
    assert B.numel() == K
    C = torch.empty((M,), device=A.device, dtype=A.dtype)
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    mv_kernel[grid](A, B, C, M, K, A.stride(0), A.stride(1), B.stride(0), BLOCK_M=BLOCK_M)
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
