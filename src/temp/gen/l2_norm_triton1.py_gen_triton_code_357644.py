
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)

    row_start_ptr = X + row_id * stride_x_row
    mask = col_offsets < N

    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0)
    rstd = tl.math.rsqrt(var + eps)
    y = x * rstd

    tl.store(Y + row_id * stride_x_row + col_offsets, y, mask=mask)


def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.stride(-1) != 1:
        x = x.contiguous()

    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    M, N = x.shape

    elem_size = x.element_size()
    BLOCK_N = min(65536 // elem_size, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise ValueError(f"N ({N}) exceeds max BLOCK_N ({BLOCK_N})")

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x,
        y,
        stride_x_row=x.stride(0),
        N=N,
        eps=eps,
        BLOCK_N=BLOCK_N,
    )

    y = y.view(original_shape)
    return y


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
