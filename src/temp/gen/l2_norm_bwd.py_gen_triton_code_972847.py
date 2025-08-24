
import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X,           # pointer to input
    DY,          # pointer to output grad
    DX,          # pointer to input grad
    stride_x_row,   # stride between rows
    N,    # number of columns
    eps,     # small epsilon
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    base_x = X + row * stride_x_row
    base_dy = DY + row * stride_x_row
    base_dx = DX + row * stride_x_row

    mask = cols < N
    x = tl.load(base_x + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(base_dy + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0)
    rstd = 1.0 / tl.sqrt(var + eps)

    coef = tl.sum(dy * x) * (1.0 / (var + eps)) * rstd
    dx = dy * rstd - coef * x

    tl.store(base_dx + cols, dx, mask=mask)

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-5):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])

    if x.stride(-1) != 1:
        x = x.contiguous()
    if dy.stride(-1) != 1:
        dy = dy.contiguous()

    assert x.shape == dy.shape
    M, N = x.shape
    dx = torch.empty_like(x)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("Feature dimension too large for L2 backward kernel.")

    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        x,
        dy,
        dx,
        x.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )
    return dx.reshape(x_shape_og)


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
