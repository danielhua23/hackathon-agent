
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256,
                       'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128,
                       'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128,
                       'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bq, stride_bs, stride_bz,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    IS_EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    k_start = pid_k * tl.cdiv(K, SPLIT_K)
    k_end   = min((pid_k + 1) * tl.cdiv(K, SPLIT_K), K)

    offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + (offs_k // 2)[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    group_size = tl.constexpr(32)
    for k in range(k_start, k_end, BLOCK_SIZE_K):
        k_valid = k + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_valid[None, :] < K), other=0.0)

        qoffs = k_valid // group_size
        shift = ((k_valid % group_size) & 1) * 4
        mask = (k_valid < K)[:, None]

        packed = tl.load(B + (k_valid // 2)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                         mask=mask & (offs_n[None, :] < N), other=0)
        packed = packed.to(tl.int32)

        scale_ptrs = B + stride_bq + qoffs[:, None] * stride_bs
        zero_ptrs  = B + stride_bq + qoffs[:, None] * stride_bz
        scale = tl.load(scale_ptrs, mask=mask, other=0.0)
        zero  = tl.load(zero_ptrs,  mask=mask, other=0.0)

        q = ((packed >> shift[:, None]) & 0xF).to(tl.float32)
        b = (q - zero) * scale
        accum += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accum, mask=mask)
    else:
        tl.atomic_add(c_ptrs, accum, mask=mask)

def matmul_dequantize_int4_s2(
    x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
    split_k: int = 1
) -> torch.Tensor:
    B, M, K = x.shape
    K_packed = w.shape[0]
    N = w.shape[1]
    assert K_packed == K // 2, f"Packed shape {K_packed} must equal K//2={K//2}"
    assert w.dtype == torch.int32
    c = torch.empty((B, M, N), dtype=x.dtype, device=x.device)

    total_M = B * M
    grid = lambda META: (
        triton.cdiv(total_M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        split_k
    )

    aux = torch.empty((2 * (scales.numel() + zeros.numel()),), dtype=torch.float32, device=w.device)
    stride_bq = w.numel() * 4
    stride_bs = scales.stride(-1) if scales.dim() >= 1 else 1
    stride_bz = zeros.stride(-1)  if zeros.dim()  >= 1 else 1

    matmul_kernel[grid](
        x.view(-1, K), w, c.view(-1, N),
        total_M, N, K,
        x.stride(-2) if x.dim() >= 2 else K,
        x.stride(-1) if x.dim() >= 1 else 1,
        w.stride(-2),
        w.stride(-1),
        c.stride(-2) if c.dim() >= 2 else N,
        c.stride(-1) if c.dim() >= 1 else 1,
        stride_bq, stride_bs, stride_bz,
        SPLIT_K=split_k,
        GROUP_SIZE_M=8,
    )
    return c

def quantize_int4(weights: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weights.dim() == 2, "weights must be 2-D (K, N)"
    K, N = weights.shape
    assert K % group_size == 0, f"K={K} must divide group_size={group_size}"
    num_groups = K // group_size
    flat = weights.to(torch.float32).view(num_groups, group_size, N)

    mn, mx = flat.aminmax(dim=1, keepdim=True)
    scale = (mx - mn) / 15.0
    scale = torch.where(scale == 0, 1.0, scale)
    zero  = -mn / scale

    q = ((flat / scale + zero + 0.5).floor()).clamp(0, 15)
    q = q.view(num_groups * group_size, N)

    q_low  = q[:q.shape[0]//2]
    q_high = q[q.shape[0]//2:]
    packed = (q_low & 0xF) | ((q_high & 0xF) << 4)
    packed = packed.view(K // 2, N).to(torch.int32)

    scale = scale.squeeze(-2).squeeze(-1)
    zero  = zero.squeeze(-2).squeeze(-1)

    return packed, scale, zero

def unpack_int4(w: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    assert w.dim() == 2 and w.dtype == torch.int32
    K_half, N = w.shape
    K = K_half * 2
    num_groups = K // group_size
    assert scales.shape[-1] == num_groups
    assert zeros.shape[-1]  == num_groups

    b0 = (w & 0xF).float()
    b1 = ((w >> 4) & 0xF).float()

    q = torch.stack([b0, b1], dim=-1).view(K, N)
    scales = scales.view(-1, 1).repeat(1, group_size).view(-1, 1)
    zeros  = zeros.view(-1, 1).repeat(1, group_size).view(-1, 1)
    unpacked = (q - zeros[:K]) * scales[:K]
    return unpacked.view(K, N).to(torch.float16)


##################################################################################################################################################





def test_correct_int4_s2(M=32, K=4096, N=4096):

    group_size = 128

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)

    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    int_b, b_scale, b_zero_point, _ = quantize_int4(b, group_size=group_size)

    

    # Test case

    triton_output = matmul_dequantize_int4_s2(a, int_b, b_scale, b_zero_point, group_size)

    

    results = {

        "test_case_1": triton_output

    }

    

    return results



result_gold = test_correct_int4_s2()
