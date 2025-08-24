
import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX,
    M, N,
    eps,
    stride_x_row,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    offsets_n = tl.arange(0, BLOCK_N)
    mask_n = offsets_n < N

    offsets_x = row_id * stride_x_row + offsets_n
    x = tl.load(X + offsets_x, mask=mask_n, other=0.0).to(tl.float32)
    dy = tl.load(DY + offsets_x, mask=mask_n, other=0.0).to(tl.float32)

    squares = x * x
    var = tl.sum(tl.where(mask_n, squares, 0.0), axis=0) / N
    rstd = tl.math.rsqrt(var + eps)

    dot = tl.sum(tl.where(mask_n, dy * x, 0.0), axis=0)

    dx = dy * rstd - dot * (1.0 / (var + eps)) * rstd * x

    tl.store(DX + offsets_x, dx, mask=mask_n)

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    original_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    dy = dy.reshape(-1, dy.size(-1)).contiguous()
    x = x.contiguous()
    M, N = x.shape
    dx = torch.empty_like(x)

    BLOCK_N = triton.next_power_of_2(N)
    if N > BLOCK_N:
        raise ValueError(f"N ({N}) exceeds maximum BLOCK_N ({BLOCK_N})")

    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        x, dy, dx,
        M, N,
        eps,
        x.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return dx.view(*original_shape)

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
