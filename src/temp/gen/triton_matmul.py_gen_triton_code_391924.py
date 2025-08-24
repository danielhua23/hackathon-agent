
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # fmt: off
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    AXIS_WM: tl.constexpr, AXIS_WN: tl.constexpr,
    IS_EVEN_K: tl.constexpr,
    DTYPE: tl.constexpr,
    # fmt: on
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not IS_EVEN_K:
            k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=k_mask, other=0.0)
            b = tl.load(b_ptrs, mask=k_mask.T, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        a = a.to(DTYPE)
        b = b.to(DTYPE)
        accumulator += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # assumes accum is always fp32
    c = accumulator.to(DTYPE)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
    assert a.dim() == 2, f"Expected 2D input tensor a, got {a.dim()}D"
    assert b.dim() == 2, f"Expected 2D input tensor b, got {b.dim()}D"
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible inner dimension for matmul: {K} vs {K_}"

    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        assert c.shape == (M, N), f"Output tensor has wrong shape: {c.shape}, expected {(M, N)}"

    # constants depending on dtype
    if a.dtype == torch.float16:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 256
        BLOCK_SIZE_K = 64
        num_stages = 2
        num_warps = 8
    else:
        # float32
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 256
        BLOCK_SIZE_K = 32
        num_stages = 2
        num_warps = 4

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    # fmt: off
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        AXIS_WM=0, AXIS_WN=1,
        IS_EVEN_K=(K % BLOCK_SIZE_K == 0),
        DTYPE=tl.float32 if a.dtype == torch.float32 else tl.float16,
        num_stages=num_stages, num_warps=num_warps,
    )
    # fmt: on

    return c


##################################################################################################################################################





import torch



# Test for matmul

def test_matmul():

    results = {}

    M, K, N = 256, 128, 256



    # Test case 1: torch.float16

    a = torch.randn((M, K), dtype=torch.float16, device='cuda')

    b = torch.randn((K, N), dtype=torch.float16, device='cuda')

    c = matmul(a, b)

    results['test_case_1'] = c



    return results



# Run all tests

result_gold = test_matmul()