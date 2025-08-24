import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_stages=2, num_warps=8)], key=['M', 'N'])
@triton.jit
def mv_kernel(A, B, C, M, N, stride_am, stride_an, stride_b, stride_c, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    offs_ms = offs_m.to(tl.int32)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_ns = offs_n.to(tl.int32)
        a_ptrs = A + offs_ms[:, None] * stride_am + offs_ns[None, :] * stride_an
        b_ptrs = B + offs_ns * stride_b
        a_block = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        b_block = tl.load(b_ptrs, mask=mask_n, other=0.0)
        acc += tl.sum(a_block * b_block, axis=1)
    out_ptrs = C + offs_ms * stride_c
    tl.store(out_ptrs, acc.to(C.dtype.element_ty), mask=mask_m)

def mv(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor=None) -> torch.Tensor:
    assert A.dim() == 2
    assert B.dim() == 1
    M, N = A.shape
    assert B.numel() == N
    if C is None:
        C = torch.empty((M,), dtype=A.dtype, device=A.device)
    else:
        assert C.shape == (M,)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    mv_kernel[grid](A, B, C, M, N, A.stride(0), A.stride(1), B.stride(0), C.stride(0))
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
