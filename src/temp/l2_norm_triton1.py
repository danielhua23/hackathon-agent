

import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}),
        triton.Config({'BLOCK_N': 128}),
        triton.Config({'BLOCK_N': 256}),
        triton.Config({'BLOCK_N': 512}),
        triton.Config({'BLOCK_N': 1024}),
    ],
    key=['N'],
)
@triton.jit
def _l2_norm_fwd_1pass_kernel(X_ptr, Y_ptr, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    
    X_row_ptr = X_ptr + row_idx * stride_x_row
    
    x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
    x_sq = x * x
    var = tl.sum(x_sq, axis=0)
    
    rstd = tl.rsqrt(var + eps)
    
    y = x * rstd
    tl.store(Y_ptr + row_idx * stride_x_row + col_offsets, y, mask=mask)

def _l2_norm_fwd(x):
    original_shape = x.shape
    x = x.view(-1, original_shape[-1])
    x = x.contiguous()
    
    M, N = x.shape
    y = torch.empty_like(x)
    
    # Calculate BLOCK_N: must not exceed 64KB
    element_size = x.element_size()
    BLOCK_N = min(8192 // element_size, N)
    
    if N > BLOCK_N:
        raise RuntimeError(f"Feature dimension {N} exceeds maximum BLOCK_N {BLOCK_N}")
    
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y,
        x.stride(0),
        N,
        1e-6,
        BLOCK_N=BLOCK_N
    )
    
    return y.view(original_shape)

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
