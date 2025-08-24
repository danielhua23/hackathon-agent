import torch
import triton
import triton.language as tl

@triton.jit
def mv_kernel(A, B, C, stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_cm: tl.constexpr, N: int, M: int, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_n = tl.program_id(0)
    start_n = pid_n * BLOCK_N
    offsets_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < N
    accum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start_m in range(0, M, BLOCK_M):
        offsets_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offsets_m < M
        offs_a = A + (offsets_n[:, None] * stride_am + offsets_m[None, :] * stride_ak)
        tile_a = tl.load(offs_a, mask=mask_n[:, None] & mask_m[None, :], other=0.0)
        offs_b = B + offsets_m * stride_bk
        seg_b = tl.load(offs_b, mask=mask_m, other=0.0)
        accum += tl.sum(tile_a * seg_b[None, :], axis=1)
    offs_c = C + offsets_n * stride_cm
    tl.store(offs_c, accum, mask=mask_n)

def mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dtype == B.dtype
    assert A.size(1) == B.size(0)
    N, M = A.shape
    C = torch.empty((N,), dtype=A.dtype, device=A.device)
    BLOCK_N = 128
    BLOCK_M = 64
    grid = (triton.cdiv(N, BLOCK_N),)
    mv_kernel[grid](A, B, C, stride_am=A.stride(0), stride_ak=A.stride(1), stride_bk=B.stride(0), stride_cm=C.stride(0), N=N, M=M, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M)
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
