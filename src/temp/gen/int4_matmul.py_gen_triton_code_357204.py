
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# INT4 matmul kernel (tiling: M tile, N tile, K tile, SPLIT-K capable)
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    x_ptr, qw_ptr, sc_ptr, zp_ptr, o_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_scg, stride_scn,
    stride_zpg, stride_zpn,
    stride_om, stride_on,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

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
    offs_k = pid_k * BLOCK_SIZE_K * SPLIT_K + tl.arange(0, BLOCK_SIZE_K * SPLIT_K)

    mask_k = offs_k < K
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, K, BLOCK_SIZE_K * SPLIT_K):
        current_offs_k = k + tl.arange(0, BLOCK_SIZE_K * SPLIT_K)
        mask_kk = current_offs_k < K

        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm +
                          current_offs_k[None, :] * stride_xk)
        x_blk = tl.load(x_ptrs, mask=mask_m[:, None] & mask_kk[None, :], other=0.0)

        qw_ptrs = qw_ptr + ((current_offs_k[None, :] // 8) * stride_qwk +
                            offs_n[:, None] * stride_qwn)
        qw_blk = tl.load(qw_ptrs, mask=mask_kk[None, :] & mask_n[:, None], other=0)

        grp_idx = (current_offs_k // group_size)
        sc_ptrs = sc_ptr + grp_idx * stride_scg + offs_n[None, :] * stride_scn
        sc_blk = tl.load(sc_ptrs, mask=mask_n[None, :], other=0.0)

        zp_ptrs = zp_ptr + grp_idx * stride_zpg + (offs_n[None, :] // 8) * stride_zpn
        zp_blk = tl.load(zp_ptrs, mask=mask_n[None, :], other=0.0)

        shifts = (current_offs_k % 8) * 4
        int4s = (qw_blk >> shifts[None, :]) & 0xF
        zp_shifts = (offs_n[None, :] % 8) * 4
        zp_int4 = (zp_blk >> zp_shifts) & 0xF
        fp_blk = (int4s.to(tl.float32) - zp_int4.to(tl.float32)) * sc_blk.to(tl.float32)

        acc += tl.dot(x_blk.to(tl.float16), fp_blk.to(tl.float16)).to(tl.float32)

    c_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_mn = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, acc, mask=mask_mn)
    else:
        tl.store(c_ptrs, acc.astype(tl.float16), mask=mask_mn)


# ------------------------------------------------------------------
# Wrapper for tensor-packed int4 inference
# ------------------------------------------------------------------
def matmul_dequantize_int4_s2(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    assert x.dim() == 2
    assert qweight.dim() == 2
    assert scale.dim() == 2
    assert zero_point.dim() == 2
    M, K = x.shape
    K8, N = qweight.shape
    assert K == K8 * 8
    assert group_size > 0
    x = x.contiguous()
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    matmul_kernel[
        lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            1,
        )
    ](
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


# ------------------------------------------------------------------
# Quantization helpers
# ------------------------------------------------------------------
def quantize_int4(x: torch.Tensor, group_size: int = 128):
    """
    Returns 3 tensors:
      packed_quant (int32), scale (float), zero_point (float)
    """
    orig_shape = x.shape
    x = x.view(-1, orig_shape[-1])
    K, N = x.shape

    if K % group_size:
        K_pad = (K + group_size - 1) // group_size * group_size
        x = torch.nn.functional.pad(x, (0, 0, 0, K_pad - K))
    else:
        K_pad = K

    x = x.view(-1, group_size)
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_max = torch.max(x_max, x_min + 1e-7)
    scale = (x_max - x_min) / 15.0
    zero_point = torch.round(-x_min / scale)

    q = torch.clamp(torch.round(x / scale + zero_point), 0, 15).to(torch.int32)

    q_f = q.view(K_pad, N)
    int32_q = torch.zeros((K_pad // 8, N), dtype=torch.int32, device=x.device)
    for shift in range(8):
        int32_q |= q_f[shift::8, :] << (shift * 4)

    int32_q = int32_q.view(*orig_shape[:-1], N // 8)
    scale = scale.view(orig_shape[0] // group_size, orig_shape[-1])
    zero_point = zero_point.view(orig_shape[0] // group_size, orig_shape[-1])

    return int32_q, scale, zero_point


def unpack_int4(packed: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, group_size: int = 128):
    """
    Decompress back to fp32 for testing.
    """
    K8, N = packed.shape
    K = K8 * 8
    fp = torch.zeros((K, N), dtype=torch.float32, device=packed.device)
    for shift in range(8):
        fp[shift::8, :] = (packed >> (shift * 4)) & 0xF
    fp = fp.view(-1, N)
    scale_rs = scale.view(-1, N)
    zp_rs = zero_point.view(-1, N)
    fp = fp.to(torch.float32)
    scale_rs = scale_rs.to(torch.float32)
    zp_rs = zp_rs.to(torch.float32)
    fp = (fp - zp_rs) * scale_rs
    return fp.view(K, N)


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
