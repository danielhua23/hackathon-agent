\n\nimport triton\nimport triton.language as tl\nimport torch\n\n@triton.jit\ndef _l2_norm_bwd_kernel(X, DY, DX, M, N, stride_x_row, stride_dy_row, stride_dx_row, eps, BLOCK_N: tl.constexpr):\n    # Changed: We now EXPECT BLOCK_N to be 16 or 32 (vectorized block size vectorized loads/stores\n    # and use proper masking regardless of actual N)\n    \n    # Locate which row this program (PID) will process\n    row_idx = tl.program_id(0)\n    if row_idx >= M:\n        return\n\n    # Offset pointers for the current row\n    x_row_ptr  = X  + row_idx * stride_x_row\n    dy_row_ptr = DY + row_idx * stride_dy_row\n    dx_row_ptr = DX + row_idx * stride_dx_row\n\n    # Gather data for this row into registers.  The mask guarantees we do not \n    # Read/write past the cache line even when N < BLOCK_N\n    cols = tl.arange(0, BLOCK_N)\n    mask = cols < N\n    x  = tl.load(x_row_ptr  + cols, mask=mask, other=0.0)\n    dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0)\n\n    # Compute squared input samplewise\n    x_sq = x * x\n    # Row-wise variance followed by unbiased correction trick via `/(N)` — here ignoring Bessel correction.\n    var = tl.sum(x_sq, axis=0) / N\n    rstd = tl.rsqrt(var + eps)\n\n    # Core backward formula\n    dot = tl.sum(dy * x, axis=0)\n    dx = dy * rstd - dot * (1.0 / (var + eps)) * rstd * x\n\n    # Store dx values for this row (mask keeps vectorized store compliant)\n    tl.store(dx_row_ptr + cols, dx, mask=mask)\n\ndef _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-5):
    """
    Computes the gradient of the input tensor `x` with respect to the loss given the upstream gradient `dy`.

    Parameters
    ----------
    x : PyTorch tensor
        Input tensor of shape (*, N) where L2 norm is applied along the last dimension.
    dy : PyTorch tensor
        Upstream gradient of same shape as `x`.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Gradient tensor `dx` of same shape as `x`.
    """
    # Flatten leading dimensions to process by row
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1]).contiguous()
    dy = dy.view(-1, dy.shape[-1]).contiguous()

    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)
    if N > BLOCK_N:
        raise ValueError(
            f"Feature dimension too large for tiled reduce: {N} > {BLOCK_N}"
        )

    # Allocate output gradient tensor
    dx = torch.empty_like(x)

    # Launch Triton grid
    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        x, dy, dx,
        M, N,
        x.stride(0), dy.stride(0), dx.stride(0),
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    # Reshape back to original shape
    return dx.view(orig_shape)\n    """\n    Computes the gradient of the input tensor `x` with respect to the loss given the upstream gradient `dy`.\n\n    Parameters\n    ----------\n    x : PyTorch tensor\n        Input tensor of shape (*, N) where L2 norm is applied along the last dimension.\n    dy : PyTorch tensor\n        Upstream gradient of same shape as `x`.\n    eps : float\n        Small value to avoid division by zero.\n\n    Returns\n    -------\n    torch.Tensor\n        Gradient tensor `dx` of same shape as `x`.\n    """\n    # Flatten leading dimensions to process by row\n    orig_shape = x.shape\n    x = x.view(-1, x.shape[-1]).contiguous()\n    dy = dy.view(-1, dy.shape[-1]).contiguous()\n\n    M, N = x.shape\n    BLOCK_N = 32               # Changed to fixed vectorized tile (32 or 16) independent of N\n    # if N > BLOCK_N:          # Removed constraint since we mask vectorized ops\n    #     raise ValueError( ... )\n\n    # Allocate output gradient tensor\n    dx = torch.empty_like(x)\n\n    # Launch Triton grid\n    grid = (M,)\n    _l2_norm_bwd_kernel[grid](\n        x, dy, dx,\n        M, N,\n        x.stride(0), dy.stride(0), dx.stride(0),\n        eps,\n        BLOCK_N=BLOCK_N,\n        num_warps=4,\n    )\n\n    # Reshape back to original shape\n    return dx.view(orig_shape)\n\n# Test the backward L2 normalization\ndef test_l2_norm_bwd():\n    results = {}\n    \n    # Test case 1: Default case\n    x = torch.randn(4, 8, device='cuda', dtype=torch.float32)\n    dy = torch.randn(4, 8, device='cuda', dtype=torch.float32)\n    dx = _l2_norm_bwd(x, dy)\n    results['test_case_1'] = dx\n\n    # Test case 2: Different shape\n    x = torch.randn(2, 16, device='cuda', dtype=torch.float32)\n    dy = torch.randn(2, 16, device='cuda', dtype=torch.float32)\n    dx = _l2_norm_bwd(x, dy)\n    results['test_case_2'] = dx\n\n    # Test case 3: Larger tensor\n    x = torch.randn(8, 8, device='cuda', dtype=torch.float32)\n    dy = torch.randn(8, 8, device='cuda', dtype=torch.float32)\n    dx = _l2_norm_bwd(x, dy)\n    results['test_case_3'] = dx\n\n    # Test case 4: Edge case with small tensor\n    x = torch.randn(1, 8, device='cuda', dtype=torch.float32)\n    dy = torch.randn(1, 8, device='cuda', dtype=torch.float32)\n    dx = _l2_norm_bwd(x, dy)\n    results['test_case_4'] = dx\n\n    return results\n\n# Run the tests\nresult_gold = test_l2_norm_bwd()\n
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
