import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(X, DY, DX, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride_x_row
    DY += row * stride_x_row
    DX += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    dy = tl.where(cols < N, dy, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    dx = dy * rstd - tl.sum(dy * x) * (1.0 / (var + eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float=1e-05):
    x_shape_orig = x.shape
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, x.shape[-1])
    x = x.contiguous()
    dy = dy.contiguous()
    M, N = x.shape
    dx = torch.empty_like(x)
    MAX_FUSED = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError('Feature dim too large for Triton kernel.')
    grid = (M,)
    _l2_norm_bwd_kernel[grid](x, dy, dx, x.stride(0), N, eps, BLOCK_N=BLOCK_N)
    return dx.reshape(x_shape_orig)

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
