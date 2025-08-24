import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    row_start = pid_m * stride_x_row
    _sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x_ptrs = X + row_start + cols
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        _sum += x_vals * x_vals
    var = tl.sum(_sum, axis=0)
    rstd = tl.math.rsqrt(var + eps)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x_ptrs = X + row_start + cols
        y_ptrs = Y + row_start + cols
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        y_vals = x_vals * rstd
        tl.store(y_ptrs, y_vals, mask=mask)

def _l2_norm_fwd(x: torch.Tensor, eps: float=1e-06):
    x = x.contiguous()
    shape = x.shape
    x = x.view(-1, shape[-1])
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_N = min(triton.next_power_of_2(N), 1 << 16)
    assert N <= BLOCK_N, 'Feature dimension N must not exceed BLOCK_N (64KB limit)'
    _l2_norm_fwd_1pass_kernel[M,](x, y, stride_x_row=x.stride(0), N=N, eps=eps, BLOCK_N=BLOCK_N)
    return y.view(*shape)

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
