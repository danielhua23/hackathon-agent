
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)

    # Row pointer
    row_start_ptr = X + row_id * stride_x_row

    # Initialize accumulators
    var = tl.zeros([BLOCK_N], dtype=tl.float32)
    mask = col_offsets < N

    # Load data
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute variance (sum of squares)
    var = x * x
    var = tl.sum(var, axis=0)

    # Compute rstd
    rstd = tl.math.rsqrt(var + eps)

    # Normalize and store
    y = x * rstd
    tl.store(Y + row_id * stride_x_row + col_offsets, y, mask=mask)


def _l2_norm_fwd(x: torch.Tensor, eps: float) -> torch.Tensor:
    # Ensure contiguous
    if x.stride(-1) != 1:
        x = x.contiguous()

    # Flatten to 2D (M, N)
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    M, N = x.shape

    # Compute BLOCK_N
    elem_size = x.element_size()
    BLOCK_N = 2 ** (16 - elem_size.bit_length())  # Max 64KB / elem_size
    BLOCK_N = min(BLOCK_N, triton.next_power_of_2(N))

    if N > BLOCK_N:
        raise ValueError(f"N ({N}) exceeds max BLOCK_N ({BLOCK_N})")

    # Prepare output
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Launch kernel
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y,
        stride_x_row=x.stride(0),
        N=N,
        eps=eps,
        BLOCK_N=BLOCK_N,
    )

    # Reshape back
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
