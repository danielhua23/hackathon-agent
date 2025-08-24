
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,
    Y,
    stride_x_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0)
    rstd = tl.math.rsqrt(var + eps)
    mask = cols < N
    y = x * rstd
    tl.store(Y + cols, y, mask=mask)


def _l2_norm_fwd(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    original_shape = x.shape
    x = x.reshape(-1, original_shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    y = torch.empty_like(x)

    M, N = x.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("L2 norm Triton kernel requires N <= BLOCK_N.")

    _l2_norm_fwd_1pass_kernel[(M,)](
        x, y,
        x.stride(0),
        N,
        eps,
        BLOCK_N=BLOCK_N
    )

    return y.reshape(original_shape)

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
