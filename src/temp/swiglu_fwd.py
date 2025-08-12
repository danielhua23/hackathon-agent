
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
        triton.Config({'BLOCK_N': 128}),
        triton.Config({'BLOCK_N': 256}),
        triton.Config({'BLOCK_N': 512}),
        triton.Config({'BLOCK_N': 1024}),
    ],
    key=['N_half'],
)
@triton.jit
def _swiglu_fwd_kernel(
    XY, OUT,
    stride_row, N_half,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N

    XY += row * stride_row
    OUT += row * stride_row

    cols = start_col + tl.arange(0, BLOCK_N)
    mask = cols < N_half

    x_idx = cols
    y_idx = cols + N_half
    x = tl.load(XY + x_idx, mask=mask, other=0.0)
    y = tl.load(XY + y_idx, mask=mask, other=0.0)

    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x * y

    tl.store(OUT + cols, out, mask=mask)


def _swiglu_fwd(xy: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.view(-1, xy.shape[-1])
    M, N_tot = xy.shape
    N_half = N_tot // 2
    if out is None:
        out = torch.empty((M, N_half), dtype=xy.dtype, device=xy.device)
    else:
        out = out.view(-1, N_half)
        assert out.shape == (M, N_half), "out has wrong shape"
        assert out.stride(-1) == 1, "out must be contiguous"
    grid = lambda META: (M, triton.cdiv(N_half, META['BLOCK_N']))
    _swiglu_fwd_kernel[grid](
        xy, out,
        xy.stride(0), N_half,
    )
    return out.view(*batch_shape, N_half)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
        triton.Config({'BLOCK_N': 128}),
        triton.Config({'BLOCK_N': 256}),
        triton.Config({'BLOCK_N': 512}),
        triton.Config({'BLOCK_N': 1024}),
    ],
    key=['N_half'],
)
@triton.jit
def _swiglu_bwd_kernel(
    XY, DOUT, DXDY,
    stride_row, N_half,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N

    XY += row * stride_row
    DOUT += row * stride_row
    DXDY += row * stride_row

    cols = start_col + tl.arange(0, BLOCK_N)
    mask = cols < N_half

    x_idx = cols
    y_idx = cols + N_half

    x = tl.load(XY + x_idx, mask=mask, other=0.0)
    y = tl.load(XY + y_idx, mask=mask, other=0.0)
    dout = tl.load(DOUT + cols, mask=mask, other=0.0)

    sig = tl.sigmoid(x)
    dsig = sig * (1 - sig)
    dx = (sig + x * dsig) * y * dout
    dy = x * sig * dout

    tl.store(DXDY + x_idx, dx, mask=mask)
    tl.store(DXDY + y_idx, dy, mask=mask)


def _swiglu_bwd(
    xy: torch.Tensor,
    dout: torch.Tensor,
    dxy: torch.Tensor = None,
) -> torch.Tensor:
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.view(-1, xy.shape[-1])
    dout = dout.view(-1, dout.shape[-1])
    M, N_tot = xy.shape
    N_half = N_tot // 2
    assert dout.shape[1] == N_half, "dout must match second half of xy"
    if dxy is None:
        dxy = torch.empty_like(xy)
    else:
        dxy = dxy.view(-1, N_tot)
        assert dxy.shape == xy.shape, "dxy has wrong shape"
        assert dxy.stride(-1) == 1, "dxy must be contiguous"
    grid = lambda META: (M, triton.cdiv(N_half, META['BLOCK_N']))
    _swiglu_bwd_kernel[grid](
        xy, dout, dxy,
        xy.stride(0), N_half,
    )
    return dxy.view(*batch_shape, N_tot)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy):
        ctx.save_for_backward(xy)
        return _swiglu_fwd(xy)

    @staticmethod
    def backward(ctx, dout):
        xy, = ctx.saved_tensors
        return _swiglu_bwd(xy, dout)


swiglu = SwiGLU.apply

##################################################################################################################################################



# Test the forward function with different configurations
def test_swiglu_fwd():
    results = {}
    # Test case 1
    batch_size = 4
    ncols = 128
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_1'] = out.detach().cpu()

    # Test case 2
    batch_size = 8
    ncols = 256
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_2'] = out.detach().cpu()

    # Test case 3
    batch_size = 16
    ncols = 512
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_3'] = out.detach().cpu()

    # Test case 4
    batch_size = 32
    ncols = 1024
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_4'] = out.detach().cpu()

    return results

# Run the tests
result_gold = test_swiglu_fwd()
