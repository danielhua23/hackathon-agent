
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
    pid = tl.program_id(0)
    if pid >= M:
        return

    x_ptr = X + pid * stride_x_row
    dy_ptr = DY + pid * stride_dy_row
    dx_ptr = DX + pid * stride_dx_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x_sq = x * x
    var = tl.sum(x_sq * mask.to(tl.float32), axis=0) / tl.sum(mask.to(tl.float32), axis=0)
    rstd = tl.rsqrt(var + eps)

    term1 = dy * rstd
    term2 = tl.sum(dy * x * mask.to(tl.float32), axis=0) / (tl.sum(mask.to(tl.float32), axis=0) * (var + eps)) * rstd * x

    dx = term1 - term2

    tl.store(dx_ptr + cols, dx, mask=mask)


def _l2_norm_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    M = x.shape[0]
    N = x.shape[-1]
    BLOCK_N = triton.next_power_of_2(N)
    if N > BLOCK_N:
        raise ValueError("Feature dimension too large")

    dx = torch.empty_like(x)

    _l2_norm_bwd_kernel[(M,)](
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
