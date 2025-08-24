
import torch
import triton
import triton.language as tl

# --------------------------------------------------
# Triton kernel
# --------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr']
)
@triton.jit
def matmul_kernel(
    x_ptr, qw_ptr, sc_ptr, zp_ptr, c_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_scg, stride_scn,
    stride_zpg, stride_zpn,
    stride_cm, stride_cn,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K * SPLIT_K + tl.arange(0, BLOCK_SIZE_K * SPLIT_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        current_offs_k = k0 * BLOCK_SIZE_K * SPLIT_K + offs_k
        mask_kk = current_offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + current_offs_k[None, :] * stride_xk
        x_blk = tl.load(x_ptrs, mask=mask_m[:, None] & mask_kk[None, :], other=0.0)

        qw_ptrs = qw_ptr + (current_offs_k[:, None] // 8) * stride_qwk + offs_n[None, :] * stride_qwn
        qw_blk = tl.load(qw_ptrs, mask=mask_kk[:, None] & mask_n[None, :], other=0)

        # scale & zp indices
        g_idx = (current_offs_k // group_size)
        sc_ptrs = sc_ptr + g_idx[:, None] * stride_scg + offs_n[None, :] * stride_scn
        zp_ptrs = zp_ptr + g_idx[:, None] * stride_zpg + (offs_n[None, :] // 8) * stride_zpn

        sc = tl.load(sc_ptrs, mask=mask_kk[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        zp = tl.load(zp_ptrs, mask=mask_kk[:, None] & mask_n[None, :], other=0)

        shifts = (current_offs_k % 8) * 4
        int4_w = (qw_blk >> shifts[:, None]) & 0xF
        zp_shifts = (offs_n[None, :] % 8) * 4
        int4_zp = (zp >> zp_shifts) & 0xF
        deq_w = ((int4_w.float() - int4_zp.float()) * sc).to(tl.float16)

        acc += tl.dot(x_blk.to(tl.float16), deq_w).to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_out = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    if SPLIT_K > 1:
        tl.atomic_add(out_ptrs, acc.astype(tl.float16), mask=mask_out)
    else:
        tl.store(out_ptrs, acc.astype(tl.float16), mask=mask_out)

# --------------------------------------------------
# Wrapper
# --------------------------------------------------
def matmul_dequantize_int4_s2(x: torch.Tensor, qweight: torch.Tensor,
                              scale: torch.Tensor, zero_point: torch.Tensor,
                              group_size: int = 128) -> torch.Tensor:
    assert x.dim() == 2
    assert qweight.dim() == 2
    assert scale.dim() == 2
    assert zero_point.dim() == 2
    M, K = x.shape
    K8, N = qweight.shape
    assert K == K8 * 8
    x = x.contiguous()
    output = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )
    matmul_kernel[grid](
        x, qweight, scale, zero_point, output,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        scale.stride(0), scale.stride(1),
        zero_point.stride(0), zero_point.stride(1),
        output.stride(0), output.stride(1),
        group_size,
    )
    return output

# --------------------------------------------------
# Quantization helpers
# --------------------------------------------------
def quantize_int4(x: torch.Tensor, group_size: int = 128):
    orig_shape = x.shape
    x = x.view(-1, orig_shape[-1])
    K, N = x.shape
    if K % group_size:
        pad_k = (K + group_size - 1) // group_size * group_size
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_k - K))
        K = pad_k
    x = x.view(-1, group_size, N)
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    denom = x_max - x_min
    denom[denom.abs() < 1e-12] = 1.0
    sc = (denom) / 15.0
    zp = torch.round(-x_min / sc)
    q = torch.clamp(torch.round(x / sc + zp), 0, 15).to(torch.int32)
    qf = q.view(K, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int32, device=x.device)
    for shift in range(8):
        packed |= qf[shift::8, :] << (shift * 4)
    sc = sc.view(K // group_size, N)
    zp = zp.view(K // group_size, N)
    return packed, sc, zp

def unpack_int4(packed: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                group_size: int = 128):
    K8, N = packed.shape
    K = K8 * 8
    w = torch.empty(K, N, dtype=torch.float32, device=packed.device)
    for shift in range(8):
        w[shift::8, :] = ((packed >> (shift * 4)) & 0xF).float()
    scale1 = scale.view(-1, N)
    zp1 = zero_point.view(-1, N)
    return ((w.view(-1, group_size, N) - zp1.unsqueeze(1)) * scale1.unsqueeze(1)).view(K, N)


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
