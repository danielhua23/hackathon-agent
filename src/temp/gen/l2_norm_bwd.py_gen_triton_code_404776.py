
import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, stride_x_row, stride_dy_row, stride_dx_row, N, eps,
    BLOCK_N: tl.constexpr
):
    pid_row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x_ptrs = X + pid_row * stride_x_row + offs_n
    dy_ptrs = DY + pid_row * stride_dy_row + offs_n
    dx_ptrs = DX + pid_row * stride_dx_row + offs_n

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x) / N
    var = tl.sum((x - mean) ** 2) / N
    rstd = tl.math.rsqrt(var + eps)

    gy = tl.sum(dy * x)
    dx = dy * rstd - gy * (1.0 / (var + eps)) * rstd * x
    dx = dx.to(DX.type.element_ty)

    tl.store(dx_ptrs, dx, mask=mask)

def _l2_norm_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dx: torch.Tensor,
    N: int,
    eps: float,
):
    M = x.numel() // N
    x = x.view(M, N) if x.stride(-1) != 1 else x
    dy = dy.view(M, N) if dy.stride(-1) != 1 else dy.contiguous()
    x = x.contiguous()
    dy = dy.contiguous()
    dx = dx.view(M, N) if dx.stride(-1) != 1 else dx
    dx = dx.contiguous()

    max_block_n = triton.next_power_of_2(N)
    BLOCK_N = max_block_n
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension {N} exceeds maximum block size {BLOCK_N}")

    grid = (triton.cdiv(M, 1),)
    _l2_norm_bwd_kernel[grid](
        x, dy, dx,
        x.stride(0), dy.stride(0), dx.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N
    )
    return dx


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
