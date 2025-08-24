import torch
import triton
import triton.language as tl

@triton.jit
def mv_kernel(A_ptr, B_ptr, C_ptr, N, M, stride_am, stride_ak, stride_bk, stride_cn, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N
    offs_n = row_start + tl.arange(0, BLOCK_N)
    col_start = 0
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for col_start in range(0, M, BLOCK_M):
        offs_m = col_start + tl.arange(0, BLOCK_M)
        a_idx = A_ptr + offs_n[:, None] * stride_am + offs_m[None, :] * stride_ak
        a_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        a_vals = tl.load(a_idx, mask=a_mask, other=0.0)
        b_idx = B_ptr + offs_m * stride_bk
        b_mask = offs_m < M
        b_vals = tl.load(b_idx, mask=b_mask, other=0.0)
        acc += tl.sum(a_vals * b_vals[None, :], axis=1)
    c_idx = C_ptr + offs_n * stride_cn
    c_mask = offs_n < N
    tl.store(c_idx, acc, mask=c_mask)

def mv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 1, 'A must be 2D and B must be 1D'
    assert A.size(1) == B.size(0), 'Inner matrix dimensions must agree'
    N, M = A.shape
    C = torch.empty((N,), dtype=torch.float32, device=A.device)

    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_N']),)
    mv_kernel[grid](A, B, C, N, M, A.stride(0), A.stride(1), B.stride(0), C.stride(0), BLOCK_N=32, BLOCK_M=32)
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
