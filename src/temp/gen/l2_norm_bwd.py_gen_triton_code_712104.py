
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, M, N, eps, stride_x_row, stride_dy_row, stride_dx_row, BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = X + row * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    dy_ptrs = DY + row * stride_dy_row + cols
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)

    dx = dy * rstd - tl.sum(dy * x, axis=0) * (rstd * rstd * rstd / N) * x

    dx_ptrs = DX + row * stride_dx_row + cols
    tl.store(dx_ptrs, dx, mask=mask)


def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float):
    shape = x.shape
    x = x.view(-1, shape[-1])
    dy = dy.view(-1, shape[-1])
    M, N = x.shape

    if not x.is_contiguous():
        x = x.contiguous()
    if not dy.is_contiguous():
        dy = dy.contiguous()

    dx = torch.empty_like(x)

    BLOCK_N = triton.next_power_of_2(N)

    if N > BLOCK_N:
        raise ValueError("N exceeds BLOCK_N")

    _l2_norm_bwd_kernel[(M,)](
        x, dy, dx, M, N, eps,
        x.stride(0), dy.stride(0), dx.stride(0),
        BLOCK_N=BLOCK_N
    )

    return dx.view(shape)


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
