
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
    if row >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x_ptr = X + row * stride_x_row
    dy_ptr = DY + row * stride_dy_row
    dx_ptr = DX + row * stride_dx_row

    x = tl.load(x_ptr + offs_n, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offs_n, mask=mask, other=0.0)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)
    sum_dy_x = tl.sum(dy * x, axis=0)
    dx = dy * rstd - sum_dy_x * (1.0 / (var + eps)) * rstd * x / N
    tl.store(dx_ptr + offs_n, dx, mask=mask)

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float):
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    dy = dy.reshape(-1, shape[-1])
    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension {N} cannot exceed {BLOCK_N}")

    dx = torch.empty_like(x)
    n_rows = M

    grid = lambda META: (n_rows,)
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
        BLOCK_N=BLOCK_N,
    )
    dx = dx.reshape(shape)
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
