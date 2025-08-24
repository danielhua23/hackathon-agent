
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(A, B, C, M, N, stride_am, stride_an, stride_bn, stride_cm, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.where(rm < M, rm, 0)
    rn = tl.where(rn < N, rn, 0)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(0, N, BLOCK_N):
        a_idx = (rm[:, None] * stride_am + (k + rn)[None, :] * stride_an)
        b_idx = (k + rn) * stride_bn
        a = tl.load(A + a_idx, mask=(k + rn)[None, :] < N, other=0).to(tl.float32)
        b = tl.load(B + b_idx, mask=(k + rn) < N, other=0).to(tl.float32)
        acc += tl.sum(a * b[None, :], axis=1)

    c_idx = rm * stride_cm
    mask = rm < M
    tl.store(C + c_idx, acc, mask=mask)


def mv(A: torch.Tensor, B: torch.Tensor):
    assert A.dim() == 2 and B.dim() == 1, "A must be 2D and B must be 1D"
    M, N = A.shape
    assert B.shape[0] == N, "Dimension mismatch between matrix and vector"
    assert A.device == B.device, "Input tensors must be on the same device"
    assert A.dtype in [torch.float16, torch.float32] and B.dtype in [torch.float16, torch.float32], "dtypes must be float16 or float32"

    C = torch.empty((M,), dtype=A.dtype, device=A.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    mv_kernel[grid](
        A, B, C, M, N,
        A.stride(0), A.stride(1),
        B.stride(0),
        C.stride(0),
        BLOCK_M=64,
        BLOCK_N=32
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
