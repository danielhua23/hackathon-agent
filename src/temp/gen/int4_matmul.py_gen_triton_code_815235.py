
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bq, stride_bs, stride_bz,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    IS_EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
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
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    start_k = 0
    if SPLIT_K > 1:
        start_k = tl.program_id(1) * tl.cdiv(K, SPLIT_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    group_size = K // (B.numel() // B.shape[0] // B.shape[1])
    q_group_size = 32

    num_groups_k = tl.cdiv(K, q_group_size)

    offs_k_p = start_k + offs_k
    for k in range(start_k, min(start_k + tl.cdiv(K, SPLIT_K), K), BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < (K - k)), other=0.0)

        idx_q = offs_k_p // q_group_size
        idx_in_q = (offs_k_p % q_group_size) // 2
        mask_even = (offs_k_p % q_group_size) % 2 == 0

        group_id = idx_q
        group_offset = group_id * stride_bq

        packed = tl.load(B + group_offset + idx_in_q[:, None] * stride_bn + offs_n[None, :] * stride_bn, mask=(idx_in_q[:, None] < (K - k) // 2) & mask_n[None, :])
        packed = packed.to(tl.int32)

        scale = tl.load(B + group_offset + stride_bs)
        zero = tl.load(B + group_offset + stride_bz)

        q0 = (packed & 0xF)
        q1 = ((packed >> 4) & 0xF)

        q0 = q0.to(tl.float32) - 8
        q1 = q1.to(tl.float32) - 8

        q = tl.where(mask_even[:, None], q0, q1)
        b = scale * q

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k_p += BLOCK_SIZE_K

    result = accumulator

    c_ptrs = C + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, result, mask=mask)
    else:
        tl.atomic_add(c_ptrs, result, mask=mask)


_configs = [
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
]

@triton.autotune(configs=_configs, key=['M', 'N', 'K'])
@triton.jit
def matmul_autotune_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bq, stride_bs, stride_bz,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    IS_EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
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
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    start_k = 0
    if SPLIT_K > 1:
        start_k = tl.program_id(1) * tl.cdiv(K, SPLIT_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    group_size = K // (B.numel() // B.shape[0] // B.shape[1])
    q_group_size = 32

    num_groups_k = tl.cdiv(K, q_group_size)

    offs_k_p = start_k + offs_k
    for k in range(start_k, min(start_k + tl.cdiv(K, SPLIT_K), K), BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < (K - k)), other=0.0)

        idx_q = offs_k_p // q_group_size
        idx_in_q = (offs_k_p % q_group_size) // 2
        mask_even = (offs_k_p % q_group_size) % 2 == 0

        group_id = idx_q
        group_offset = group_id * stride_bq

        packed = tl.load(B + group_offset + idx_in_q[:, None] * stride_bn + offs_n[None, :] * stride_bn, mask=(idx_in_q[:, None] < (K - k) // 2) & mask_n[None, :])
        packed = packed.to(tl.int32)

        scale = tl.load(B + group_offset + stride_bs)
        zero = tl.load(B + group_offset + stride_bz)

        q0 = (packed & 0xF)
        q1 = ((packed >> 4) & 0xF)

        q0 = q0.to(tl.float32) - 8
        q1 = q1.to(tl.float32) - 8

        q = tl.where(mask_even[:, None], q0, q1)
        b = scale * q

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k_p += BLOCK_SIZE_K

    result = accumulator

    c_ptrs = C + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, result, mask=mask)
    else:
        tl.atomic_add(c_ptrs, result, mask=mask)


def matmul_dequantize_int4_s2(
    x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
    split_k: int = 1
) -> torch.Tensor:
    B, M, K = x.shape
    N, K_packed = w.shape
    group_size = K // (w.numel() // w.shape[0] // w.shape[1])

    assert K_packed == K // 2, (
        f"Expected packed weight shape {K // 2}, got {K_packed}"
    )
    assert w.dtype == torch.int32

    c = torch.empty((B, M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (
        triton.cdiv(M * B, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        split_k,
    )

    matmul_autotune_kernel[grid](
        x.flatten(0, 1), w, c.flatten(0, 1),
        M * B, N, K,
        x.stride(1), x.stride(2),
        w.stride(1), w.stride(0),
        c.stride(1), c.stride(2),
        scales.stride(0) if scales.dim() > 1 else 0,
        scales.stride(0) if scales.dim() > 1 else 1,
        zeros.stride(0) if zeros.dim() > 1 else 0,
        SPLIT_K=split_k,
    )
    return c


def quantize_int4(x: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, K = x.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"

    x = x.to(torch.float32)
    x = x.view(B, -1, group_size)

    mn, mx = x.aminmax(dim=2, keepdim=True)
    scale = (mx - mn) / 15
    scale = torch.where(scale == 0, 1, scale)
    zero = -mn / scale

    xq = ((x / scale + zero + 0.5).floor()).clamp(0, 15).to(torch.int32)

    xq = (xq.view(B, -1, 4) << torch.tensor([0, 4, 8, 12], device=xq.device)).sum(2).to(torch.int32)

    scale = scale.view(B, -1)
    zero = zero.view(B, -1)

    return xq, scale, zero


def unpack_int4(w: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    B, K_packed = w.shape
    K = K_packed * 2

    scales = scales.to(torch.float32)
    zeros = zeros.to(torch.float32)

    w = w.view(B, -1)
    ws = (w[..., None] >> torch.tensor([0, 4], dtype=torch.int32, device=w.device)) & 0xF

    ws = ws.flatten(-2)

    ws = ws.view(B, -1, group_size)

    ws = (ws.to(torch.float32) - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)

    ws = ws.view(B, K).to(torch.float16)

    return ws


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
