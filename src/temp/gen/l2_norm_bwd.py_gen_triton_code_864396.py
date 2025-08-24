
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, M, N,
    stride_x_row,
    stride_x_col,
    stride_dy_row,
    stride_dy_col,
    stride_dx_row,
    stride_dx_col,
    eps,
    BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x_ptr = X + row_idx * stride_x_row + offs_n * stride_x_col
    dy_ptr = DY + row_idx * stride_dy_row + offs_n * stride_dy_col
    dx_ptr = DX + row_idx * stride_dx_row + offs_n * stride_dx_col

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr, mask=mask, other=0.0).to(tl.float32)

    x_sum = tl.sum(x * x)
    var = x_sum / N
    rstd = tl.math.rsqrt(var + eps)

    grad_x_part = tl.sum(dy * x)
    coef = (1.0 / (var + eps)) * rstd * grad_x_part
    dx = dy * rstd - coef * x

    tl.store(dx_ptr, dx.to(tl.load(x_ptr).dtype), mask=mask)


def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float):
    assert x.shape == dy.shape, "x and dy must have the same shape"
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
        dy = dy.view(-1, dy.shape[-1])
    orig_shape = x.shape

    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)
    if N > BLOCK_N:
        raise RuntimeError(f"Feature dimension {N} exceeds allowed BLOCK_N={BLOCK_N}")

    x = x.contiguous()
    dy = dy.contiguous()

    dx = torch.empty_like(x)

    grid = lambda META: (M,)
    _l2_norm_bwd_kernel[grid](
        x, dy, dx, M, N,
        x.stride(0), x.stride(1) if x.dim() == 2 else 0,
        dy.stride(0), dy.stride(1) if dy.dim() == 2 else 0,
        dx.stride(0), dx.stride(1) if dx.dim() == 2 else 0,
        eps,
        BLOCK_N=BLOCK_N
    )

    return dx.view(orig_shape)



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
