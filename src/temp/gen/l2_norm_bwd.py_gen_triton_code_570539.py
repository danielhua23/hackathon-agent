
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_bwd_kernel(X, DY, DX, M, N, stride_x_row, eps, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    x_ptrs = X + pid_m * stride_x_row + offs_n
    dy_ptrs = DY + pid_m * N + offs_n
    dx_ptrs = DX + pid_m * N + offs_n

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x) / N
    rstd = tl.math.rsqrt(var + eps)

    term = tl.sum(dy * x) * (1.0 / (var + eps)) * rstd
    dx = dy * rstd - term * x

    tl.store(dx_ptrs, dx, mask=mask)


def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-12):
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    if dy.dim() > 2:
        dy = dy.reshape(-1, dy.shape[-1])

    M = x.shape[0]
    N = x.shape[1]
    dx = torch.empty_like(x)

    if not x.is_contiguous():
        x = x.contiguous()
    if not dy.is_contiguous():
        dy = dy.contiguous()

    BLOCK_N = triton.next_power_of_2(N)
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension N ({N}) exceeds block size BLOCK_N ({BLOCK_N}).")

    _l2_norm_bwd_kernel[(M,)](
        x,
        dy,
        dx,
        M,
        N,
        x.stride(0),
        eps,
        BLOCK_N=BLOCK_N,
    )

    return dx.reshape(x.shape)


##################################################################################################################################################





import torch



# Test the backward L2 normalization

def test_l2_norm_bwd():

    results = {}

    

    # Test case 1: Default case

    x = torch.randn(4, 8, device='cuda', dtype=torch.float32)

    dy = torch.randn(4, 8, device='cuda', dtype=torch.float32)

    dx = _l2_norm_bwd(x, dy)

    results['test_case_1'] = dx



    # Test case 2: Different shape

    x = torch.randn(2, 16, device='cuda', dtype=torch.float32)

    dy = torch.randn(2, 16, device='cuda', dtype=torch.float32)

    dx = _l2_norm_bwd(x, dy)

    results['test_case_2'] = dx



    # Test case 3: Larger tensor

    x = torch.randn(8, 8, device='cuda', dtype=torch.float32)

    dy = torch.randn(8, 8, device='cuda', dtype=torch.float32)

    dx = _l2_norm_bwd(x, dy)

    results['test_case_3'] = dx



    # Test case 4: Edge case with small tensor

    x = torch.randn(1, 8, device='cuda', dtype=torch.float32)

    dy = torch.randn(1, 8, device='cuda', dtype=torch.float32)

    dx = _l2_norm_bwd(x, dy)

    results['test_case_4'] = dx



    return results



# Run the tests

result_gold = test_l2_norm_bwd()
