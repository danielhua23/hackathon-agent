
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    # program ids
    row_id = tl.program_id(0)

    # offsets
    offs_n = tl.arange(0, BLOCK_N)

    # compute normalized offset
    row_start = X + row_id * stride_x_row
    y_row_start = Y + row_id * stride_x_row

    # compute sum of squares
    var = tl.zeros([], dtype=tl.float32)
    masked_offs = offs_n < N
    for i in range(0, N, BLOCK_N):
        offs = i + offs_n
        mask = masked_offs & (offs < N)
        x_ptrs = row_start + offs  # assuming the tensor has stride = 1 in the last dimension
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        var += tl.sum(x.to(tl.float32) * x.to(tl.float32), axis=0)

    # Compute rstd
    rstd = tl.rsqrt(var + eps)

    # normalize and store
    for i in range(0, N, BLOCK_N):
        offs = i + offs_n
        mask = masked_offs & (offs < N)
        x_ptrs = row_start + offs
        y_ptrs = y_row_start + offs
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_normed = x.to(tl.float32) * rstd
        tl.store(y_ptrs, x_normed.to(Y.type.element_ty), mask=mask)


def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-5):
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    M, N = x.shape
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)

    element_size = x.element_size()
    max_block_size = 65536 // element_size
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > max_block_size:
        BLOCK_N = triton.next_power_of_2(triton.cdiv(max_block_size, 8))
    assert N <= BLOCK_N, "Feature dimension exceeds the max block size"

    _l2_norm_fwd_1pass_kernel[(M,)](
        x, y,
        x.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N
    )
    return y.view(original_shape)

##################################################################################################################################################



import torch

# Test the forward L2 normalization
def test_l2_norm_fwd():
    results = {}
    
    # Test case 1
    x1 = torch.randn(4, 8, device='cuda', dtype=torch.float32)
    y1 = _l2_norm_fwd(x1)
    results['test_case_1'] = y1

    # Test case 2: Different batch size
    x2 = torch.randn(2, 8, device='cuda', dtype=torch.float32)
    y2 = _l2_norm_fwd(x2)
    results['test_case_2'] = y2

    # Test case 3: Different feature size
    x3 = torch.randn(4, 4, device='cuda', dtype=torch.float32)
    y3 = _l2_norm_fwd(x3)
    results['test_case_3'] = y3

    # Test case 4: Larger tensor
    x4 = torch.randn(8, 8, device='cuda', dtype=torch.float32)
    y4 = _l2_norm_fwd(x4)
    results['test_case_4'] = y4

    return results

result_gold = test_l2_norm_fwd()
