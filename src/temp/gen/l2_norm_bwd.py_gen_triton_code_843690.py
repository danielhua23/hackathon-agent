
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX,
    M, N, eps,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x_ptrs = X + row_id * stride_x_row + offs_n
    dy_ptrs = DY + row_id * stride_dy_row + offs_n
    dx_ptrs = DX + row_id * stride_dx_row + offs_n

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) * (x - mean), axis=0) / N
    rstd = 1.0 / tl.math.sqrt(var + eps)

    dx = dy * rstd - tl.sum(dy * x, axis=0) * (1.0 / (var + eps)) * rstd * x
    tl.store(dx_ptrs, dx, mask=mask)


def _l2_norm_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    eps: float,
):
    shape = list(x.shape)
    x = x.view(-1, shape[-1])
    dy = dy.view(-1, shape[-1])
    assert x.shape == dy.shape, "x and dy must have the same shape"
    M, N = x.shape
    dx = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    if N > BLOCK_N:
        raise RuntimeError(f"This l2 norm does not support feature dim >= {BLOCK_N}.")

    _l2_norm_bwd_kernel[(M,)](
        x, dy, dx,
        M, N, eps,
        x.stride(0),
        dy.stride(0),
        dx.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return dx.view(*shape)


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
