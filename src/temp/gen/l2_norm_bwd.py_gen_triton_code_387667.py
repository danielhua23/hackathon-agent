
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_bwd_kernel(
    X,
    DY,
    DX,
    M,
    N,
    eps,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Compute the base pointers for this row
    x_ptrs = X + row * stride_x_row + cols
    dy_ptrs = DY + row * stride_dy_row + cols
    dx_ptrs = DX + row * stride_dx_row + cols

    # Load X and DY for this row
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0)

    # Compute variance and reciprocal standard deviation
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)
    
    # Compute the gradient contribution
    term1 = dy * rstd
    term2 = tl.sum(dy * x, axis=0) * rstd / (var + eps) * x
    
    dx = term1 - term2
    
    # Store the result
    tl.store(dx_ptrs, dx, mask=mask)


def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float) -> torch.Tensor:
    # Ensure inputs are contiguous
    x = x.contiguous()
    dy = dy.contiguous()
    
    # Get original shape
    original_shape = x.shape
    
    # Reshape to 2D if necessary
    M = x.numel() // original_shape[-1]
    N = original_shape[-1]
    x = x.view(M, N)
    dy = dy.view(M, N)
    
    # Determine optimal block size
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    
    # Ensure N doesn't exceed BLOCK_N
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension {N} is too large")
    
    # Allocate output
    dx = torch.empty_like(x)
    
    # Configure kernel launch
    kernel_args = {
        'BLOCK_N': BLOCK_N,
    }

    # Launch kernel
    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        x,
        dy,
        dx,
        M,
        N,
        eps,
        x.stride(0),
        dy.stride(0),
        dx.stride(0),
        **kernel_args
    )
    
    # Restore original shape
    dx = dx.view(original_shape)
    
    return dx


##################################################################################################################################################





import torch



# Test the backward L2 normalization

def test_l2_norm_bwd():

    results = {}

    

    # Test case 1: Default case

    x = torch.randn(4, 8, device='cuda', dtype=torch.float32)

    dy = torch.randn(4, 8, device='cuda', dtype=torch.float32)

    dx = _l2_norm_bwd(x, dy)

    results['test_case_1'] = dx



    # Test case 2: Different shape

    x = torch.randn(2, 16, device='cuda', dtype=torch.float32)

    dy = torch.randn(2, 16, device='cuda', dtype=torch.float32)

    dx = _l2_norm_bwd(x, dy)

    results['test_case_2'] = dx



    # Test case 3: Larger tensor

    x = torch.randn(8, 8, device='cuda', dtype=torch.float32)

    dy = torch.randn(8, 8, device='cuda', dtype=torch.float32)

    dx = _l2_norm_bwd(x, dy)

    results['test_case_3'] = dx



    # Test case 4: Edge case with small tensor

    x = torch.randn(1, 8, device='cuda', dtype=torch.float32)

    dy = torch.randn(1, 8, device='cuda', dtype=torch.float32)

    dx = _l2_norm_bwd(x, dy)

    results['test_case_4'] = dx



    return results



# Run the tests

result_gold = test_l2_norm_bwd()
