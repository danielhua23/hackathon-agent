
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

    x_vals = tl.load(row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x_vals * x_vals)
    rstd = 1.0 / tl.sqrt(var + eps)

    out_vals = x_vals * rstd
    tl.store(row_out_start + col_offsets, out_vals, mask=mask)


def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-6):
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
        raise ValueError("N too large")

    grid = lambda META: (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y,
        x.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N,
    )

    return y.view(shape)
