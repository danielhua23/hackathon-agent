
import torch
import triton
import triton.language as tl

@triton.jit
def mv_kernel(A, B, C, stride_am, stride_an, stride_bn, stride_cm, N, M,
              BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_n = tl.program_id(0)

    n0 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, M, BLOCK_M):
        m_offs = k + tl.arange(0, BLOCK_M)
        mask_m = m_offs < M
        a_ptrs = A + (n0[:, None] * stride_am + m_offs[None, :] * stride_an)
        a_mask = (n0[:, None] < N) & mask_m[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B + m_offs * stride_bn
        b = tl.load(b_ptrs, mask=mask_m, other=0.0)

        acc += tl.sum(a * b, axis=1)

    c_ptrs = C + n0 * stride_cm
    c_mask = n0 < N
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)

def mv(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor = None):
    assert A.dim() == 2 and B.dim() == 1
    N, M = A.shape
    assert B.shape[0] == M
    if C is None:
        C = torch.empty((N,), dtype=A.dtype, device=A.device)

    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)

    BLOCK_N = 64
    BLOCK_M = 64

    mv_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0),
        C.stride(0),
        N, M,
        BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M
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
