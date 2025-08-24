
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptr = X + row * stride_x_row
    y_ptr = Y + row * stride_x_row

    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x)
    rstd = tl.math.rsqrt(var + eps)

    y = x * rstd
    tl.store(y_ptr + cols, y.to(Y.dtype.element_ty), mask=mask)


def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-5):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    M, N = x.shape

    y = torch.empty_like(x)

    max_fused_size = 65536 // x.element_size()
    BLOCK_N = min(max_fused_size, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(f"This L2 norm does not support feature dim >= 64KB. Got: {N}")

    _l2_norm_fwd_1pass_kernel[(M,)](
        X=x,
        Y=y,
        stride_x_row=x.stride(0),
        N=N,
        eps=eps,
        BLOCK_N=BLOCK_N,
    )

    return y.reshape(x_shape_og)


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
