
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # compute base pointers
    x_ptr = X + row_id * stride_x_row
    y_ptr = Y + row_id * stride_x_row

    # load and accumulate
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x)
    rstd = tl.math.rsqrt(var + eps)

    # normalize and store
    y = x * rstd
    tl.store(y_ptr + cols, y.to(Y.type.element_ty), mask=mask)


def _l2_norm_fwd(x: torch.Tensor, eps: float):
    if x.dim() == 3:
        B, M, N = x.shape
        x_view = x.view(B * M, N)
    elif x.dim() == 2:
        B, M, N = 1, *x.shape
        x_view = x
    else:
        raise ValueError("Unsupported input dim")

    x_contig = x_view.contiguous()
    y = torch.empty_like(x_contig)

    BLOCK_N = triton.next_power_of_2(N)
    byte_per_elem = x_contig.element_size()
    upper = (65536 // byte_per_elem)
    if BLOCK_N > upper:
        BLOCK_N = (65536 // byte_per_elem)

    assert N <= BLOCK_N, f"Feature dim {N} exceeds limit {BLOCK_N}"

    _l2_norm_fwd_1pass_kernel[(x_contig.shape[0],)](
        x_contig,
        y,
        x_contig.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )

    if x.dim() == 3:
        return y.view(B, M, N)
    return y.view(x.shape)


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
