
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = X + row_idx * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0)
    rstd = tl.math.rsqrt(var + eps)
    y = x * rstd

    y_ptrs = Y + row_idx * stride_x_row + cols
    tl.store(y_ptrs, y.to(Y.type.element_ty), mask=mask)


def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.contiguous()
    *shape, N = x.shape
    x = x.view(-1, N)
    M = x.size(0)

    element_size = x.element_size()
    max_block = (64 * 1024) // element_size
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > max_block:
        raise ValueError(f"N ({N}) exceeds block size limit ({max_block})")

    y = torch.empty_like(x)

    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y,
        x.stride(0), N, eps,
        BLOCK_N=BLOCK_N
    )

    return y.view(*shape)


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
