
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_bwd_kernel(X, DY, DX, M, N, eps, stride_x_row, stride_dy_row, stride_dx_row, BLOCK_N: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = X + row_id * stride_x_row + cols
    dy_ptrs = DY + row_id * stride_dy_row + cols
    dx_ptrs = DX + row_id * stride_dx_row + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0)

    x_fp32 = x.to(tl.float32)
    var = tl.sum(x_fp32 * x_fp32, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)

    dy_fp32 = dy.to(tl.float32)
    dot = tl.sum(dy_fp32 * x_fp32, axis=0)
    coeff = dot * (1.0 / (var + eps)) * rstd
    dx = dy_fp32 * rstd - coeff * x_fp32

    dx_out = dx.to(DX.type.element_ty)
    tl.store(dx_ptrs, dx_out, mask=mask)


def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float):
    assert x.shape == dy.shape, "Shape mismatch between x and dy"

    original_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
        dy = dy.view(-1, dy.shape[-1])

    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)
    if N > BLOCK_N:
        raise ValueError("Feature dimension N too large for BLOCK_N")

    dx = torch.empty_like(x)

    if not x.is_contiguous():
        x = x.contiguous()
    if not dy.is_contiguous():
        dy = dy.contiguous()

    stride_x_row = x.stride(0) if x.stride(-1) == 1 else x.stride(-2)
    stride_dy_row = dy.stride(0) if dy.stride(-1) == 1 else dy.stride(-2)
    stride_dx_row = dx.stride(0) if dx.stride(-1) == 1 else dx.stride(-2)

    _l2_norm_bwd_kernel[(M,)](
        x,
        dy,
        dx,
        M,
        N,
        eps,
        stride_x_row,
        stride_dy_row,
        stride_dx_row,
        BLOCK_N=BLOCK_N,
    )

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
