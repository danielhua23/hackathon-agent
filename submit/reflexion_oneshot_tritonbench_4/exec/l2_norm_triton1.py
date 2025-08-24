import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_N': 256}, num_warps=2, num_stages=1), triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=1), triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=1), triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=1)], key=['N'])
@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(tl.where(mask, x * x, 0.0))
    rstd = 1.0 / tl.sqrt(var + eps)
    out = x * rstd
    tl.store(Y + row * stride_x_row + offs, out, mask=mask)

def _l2_norm_fwd(x: torch.Tensor, eps: float=1e-06):
    shape = x.shape
    x = x.view(-1, shape[-1]).contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](x, y, x.stride(0), N, eps)
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
