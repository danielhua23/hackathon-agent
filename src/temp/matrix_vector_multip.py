
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(
    A, B, C,
    N, M,
    stride_an, stride_am,
    stride_b,
    stride_c,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k in range(0, M, BLOCK_M):
        offs_am = k + offs_m
        mask_a = (offs_n[:, None] < N) & (offs_am[None, :] < M)
        mask_b = offs_am < M
        a_ptrs = A + (offs_n[:, None] * stride_an + offs_am[None, :] * stride_am)
        b_ptrs = B + offs_am * stride_b
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.sum(a * b[None, :], axis=1)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = offs_cn < N
    c_ptrs = C + offs_cn * stride_c
    tl.store(c_ptrs, acc, mask=mask_c)


def mv(A: torch.Tensor, B: torch.Tensor):
    assert A.dim() == 2, "A must be 2-D"
    assert B.dim() == 1, "B must be 1-D"
    N, M = A.shape
    assert B.shape[0] == M, "Incompatible shapes for matrix-vector multiplication"
    C = torch.empty(N, dtype=A.dtype, device=A.device)

    BLOCK_N = 64
    BLOCK_M = 64

    grid = lambda META: (triton.cdiv(N, META['BLOCK_N']),)

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
