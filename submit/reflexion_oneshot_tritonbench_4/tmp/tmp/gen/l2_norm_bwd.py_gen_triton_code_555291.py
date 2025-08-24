import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_N': 256}, num_warps=2, num_stages=1), triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=1), triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=1), triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=1)], key=['N'])
@triton.jit
def _l2_norm_bwd_kernel(X, DY, DX, stride_x_row, stride_dy_row, stride_dx_row, N, eps, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x_ptr = X + pid * stride_x_row
    dy_ptr = DY + pid * stride_dy_row
    dx_ptr = DX + pid * stride_dx_row
    x_vals = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dy_vals = tl.load(dy_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = x_vals * x_vals
    var = tl.sum(x2, axis=0)
    r = tl.sqrt(var + eps)
    rstd = 1.0 / r
    dy_dot_x = tl.sum(dy_vals * x_vals, axis=0)
    rs3 = rstd * rstd * rstd
    dx = dy_vals * rstd - dy_dot_x * rs3 * x_vals
    tl.store(dx_ptr + cols, dx.to(DX.dtype.element_ty), mask=mask)

def _l2_norm_bwd(x, dy, eps: float=1e-05):
    assert x.shape == dy.shape
    assert x.ndim >= 2
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    M, N = x.shape
    dx = torch.empty_like(x)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(triton.next_power_of_2(N), MAX_FUSED_SIZE)
    if N > BLOCK_N:
        raise RuntimeError("This L2 norm doesn't support feature dim >= 64KB.")
    grid = lambda META: (M,)
    _l2_norm_bwd_kernel[grid](x, dy, dx, x.stride(0), dy.stride(0), dx.stride(0), N, eps)
    return dx.reshape(orig_shape)

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
