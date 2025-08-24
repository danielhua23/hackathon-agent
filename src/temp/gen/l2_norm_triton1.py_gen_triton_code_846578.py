
import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    X_ptr = X + row_id * stride_x_row + col_offsets
    x = tl.load(X_ptr, mask=mask, other=0.0)
    sum2 = x.to(tl.float32) * x.to(tl.float32)
    var = tl.sum(sum2, axis=0)
    rstd = tl.math.rsqrt(var + eps)
    out = x * rstd
    Y_ptr = Y + row_id * stride_x_row + col_offsets
    tl.store(Y_ptr, out, mask=mask)

def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.dim() == 3:
        b, m, n = x.shape
        x = x.view(-1, n)
        orig_shape = (b, m, n)
    else:
        orig_shape = x.shape
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    BLOCK_N = 1024 // x.element_size()
    if N > BLOCK_N:
        raise RuntimeError(f"Feature dimension N={N} exceeds maximum {BLOCK_N}")

    _l2_norm_fwd_1pass_kernel[(M,)](
        x,
        y,
        x.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )
    return y.view(*orig_shape)


##################################################################################################################################################





import torch



# Test the forward L2 normalization

def test_l2_norm_fwd():

    results = {}

    

    # Test case 1

    x1 = torch.randn(4, 8, device='cuda', dtype=torch.float32)

    y1 = _l2_norm_fwd(x1)

    results['test_case_1'] = y1



    # Test case 2: Different batch size

    x2 = torch.randn(2, 8, device='cuda', dtype=torch.float32)

    y2 = _l2_norm_fwd(x2)

    results['test_case_2'] = y2



    # Test case 3: Different feature size

    x3 = torch.randn(4, 4, device='cuda', dtype=torch.float32)

    y3 = _l2_norm_fwd(x3)

    results['test_case_3'] = y3



    # Test case 4: Larger tensor

    x4 = torch.randn(8, 8, device='cuda', dtype=torch.float32)

    y4 = _l2_norm_fwd(x4)

    results['test_case_4'] = y4



    return results



result_gold = test_l2_norm_fwd()
