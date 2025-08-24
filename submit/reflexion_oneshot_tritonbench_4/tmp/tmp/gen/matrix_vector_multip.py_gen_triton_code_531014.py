import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(configs=[triton.Config({'BLOCK_M': 1, 'BLOCK_K': 64}, num_warps=4, num_stages=2), triton.Config({'BLOCK_M': 2, 'BLOCK_K': 64}, num_warps=4, num_stages=2), triton.Config({'BLOCK_M': 4, 'BLOCK_K': 64}, num_warps=4, num_stages=2), triton.Config({'BLOCK_M': 4, 'BLOCK_K': 128}, num_warps=8, num_stages=2), triton.Config({'BLOCK_M': 8, 'BLOCK_K': 64}, num_warps=4, num_stages=2), triton.Config({'BLOCK_M': 8, 'BLOCK_K': 128}, num_warps=8, num_stages=2)], key=['M', 'N'])
@triton.jit
def mv_kernel(A, B, C, M, N, stride_am, stride_an, stride_b, stride_c, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_n = offs_n < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    LOOP_K_MAX: tl.constexpr = tl.cdiv(N, BLOCK_K)
    for k_off in tl.static_range(0, LOOP_K_MAX):
        offs_k = k_off * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < N
        a_ptrs = A + (offs_n[:, None] * stride_am + offs_k[None, :] * stride_an)
        b_ptrs = B + offs_k * stride_b
        a_blk = tl.load(a_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        b_blk = tl.load(b_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(a_blk * b_blk[None, :], 1)
    c_ptrs = C + offs_n * stride_c
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_n)

def mv(A: torch.Tensor, B: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 1, 'A must be 2-D and B must be 1-D'
    M, N = A.shape
    assert B.numel() == N, 'Size mismatch'
    assert A.dtype == B.dtype, 'dtype mismatch'
    if out is None:
        out = torch.empty(M, dtype=A.dtype, device=A.device)
    else:
        assert out.dtype == A.dtype and out.numel() == M, 'out mismatch'
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    mv_kernel[grid](A, B, out, M, N, A.stride(0), A.stride(1), B.stride(0), out.stride(0))
    return out

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
