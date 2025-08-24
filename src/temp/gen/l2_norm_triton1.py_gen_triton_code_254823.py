
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr
):
    row_id = tl.program_id(0)
    row_start = X + row_id * stride_x_row
    row_out_start = Y + row_id * stride_x_row

    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_vals = tl.load(row_start + col_offsets, mask=mask, other=0.0)
    acc = x_vals * x_vals
    var = tl.sum(acc)
    rstd = 1.0 / tl.sqrt(var + eps)

    out_vals = x_vals * rstd
    tl.store(row_out_start + col_offsets, out_vals, mask=mask)


def _l2_norm_fwd(x: torch.Tensor, eps: float):
    x = x.contiguous()
    shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])
    M, N = x.shape
    y = torch.empty_like(x)

    element_size = x.element_size()
    MAX_FUSED = 65536 // element_size
    BLOCK_N = min(MAX_FUSED, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise ValueError

    grid = lambda META: (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y,
        x.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )

    return y.view(shape)


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
