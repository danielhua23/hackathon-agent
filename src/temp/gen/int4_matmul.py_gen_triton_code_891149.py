
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
    x_ptr,                  # (M, K)  fp16/bf16
    qw_ptr,                 # (K//8, N) packed INT4 in INT32 (8x 4b per int32)
    sc_ptr,                 # (num_groups, N) fp16/bf16
    zp_ptr,                 # (num_groups, N) fp16/bf16
    o_ptr,                  # (M, N)  fp16/bf16
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
    SPLIT_K: tl.constexpr = 1,          # Use for split-k reduction
):
    # Program & tile coordinates
    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)       # for SPLIT_K
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Global tile spans
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # Adjust for SPLIT_K
    k_max = (pid_k + 1) * BLOCK_SIZE_K
    if k_max > K:
        k_max = K
    # Clamp inside kernel
    mask_k = offs_k < K
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers in batch offset
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    qw_ptrs = qw_ptr + ((offs_k[None, :] // 8) * stride_qwk + offs_n[:, None] * stride_qwn)
    sc_ptrs = sc_ptr + ((offs_n[:, None] // group_size) * stride_scg + offs_n[:, None] * stride_scn)
    zp_ptrs = zp_ptr + ((offs_n[:, None] // group_size) * stride_zpg + offs_n[:, None] * stride_zpn)

    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        qwp = tl.load(qw_ptrs, mask=mask_k[None, :] & mask_n[:, None], other=0)
        scales = tl.load(sc_ptrs, mask=mask_n[:, None], other=0.0)
        zps    = tl.load(zp_ptrs, mask=mask_n[:, None], other=0.0)

        # unpack 8x INT4 per int32
        local_offs = (offs_k % 8) * 4         # (BLOCK_SIZE_K,) -> 0,4,8,...,28
        q4_mask    = 0xF                      # 4 bits
        qw_int4    = (qwp >> local_offs) & q4_mask
        qw_fp      = (qw_int4.to(tl.float32) - zps) * scales

        # accumulate matmul
        accum += tl.dot(a, qw_fp)

        # advance
        x_ptrs += BLOCK_SIZE_K * stride_xk
        qw_ptrs += (BLOCK_SIZE_K // 8) * stride_qwk

    if SPLIT_K > 1:
        o_blk_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        mask_mn = mask_m[:, None] & mask_n[None, :]
        tl.atomic_add(o_blk_ptrs, accum, mask=mask_mn)
    else:
        o_blk_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        mask_mn = mask_m[:, None] & mask_n[None, :]
        tl.store(o_blk_ptrs, accum, mask=mask_mn)


# ------------------------------------------------------------------
# High-level wrapper launching quantized matmul kernel
# ------------------------------------------------------------------
def matmul_dequantize_int4_s2(x: torch.Tensor,
                              qweight_int32: torch.Tensor,
                              scale: torch.Tensor,
                              zero_point: torch.Tensor,
                              split_k: int = 1):
    """
    Launch INT4 GEMM:  x @ dequantize(qw)

    x         : (M, K) fp16/bf16
    qw        : (K//8, N) int32 packed
    scale/zp  : (num_groups, N) fp16/bf16
    Returns   : (M, N) fp16/bf16
    """
    assert x.dim() == 2
    assert qweight_int32.dim() == 2
    assert scale.dim() == 2
    assert zero_point.dim() == 2
    M, K = x.shape
    K8, N = qweight_int32.shape
    assert K == K8 * 8

    group_size = K // scale.shape[0]
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), split_k)

    matmul_kernel[grid](
        x, qweight_int32, scale, zero_point, output,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight_int32.stride(0), qweight_int32.stride(1),
        scale.stride(0), scale.stride(1),
        zero_point.stride(0), zero_point.stride(1),
        output.stride(0), output.stride(1),
        group_size,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        SPLIT_K=split_k,
    )
    if split_k > 1:
        # Reduction here, currently left to caller
        pass
    return output


# ------------------------------------------------------------------
# Quantization utilities
# ------------------------------------------------------------------
def quantize_int4(x: torch.Tensor, group_size: int = 128):
    """
    Convert fp tensor to INT4 packed (8 int4 per int32), return (qint32, scale, zero_point)
    """
    *rest, N = x.shape
    x = x.reshape(-1, N).contiguous()

    pad = (group_size - (N % group_size)) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    x = x.view(-1, group_size)

    # per-group min/max -> scale, zero-pt
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_max = torch.max(x_max, x_min + 1e-7)            # ensure non-degenerate
    scale = (x_max - x_min) / 15.0
    zero_point = (-x_min / scale)

    # quantize & clamp
    xq = torch.round(x / scale + zero_point)
    xq = xq.clamp(0, 15).to(torch.int32)

    # reshape back to packed layout
    xq = xq.view(-1)
    # pack 8 int4 into int32 (order: lowest 4 bits first)
    num_i32 = xq.numel() // 8
    qw = torch.zeros(num_i32, dtype=torch.int32, device=x.device)
    for shift in range(8):
        qw |= (xq[shift::8] << (shift * 4))

    # reshape back to original mapping
    qw = qw.view(*rest, -1)
    scale = scale.view(*rest, -1)
    zero_point = zero_point.view(*rest, -1)
    return qw, scale, zero_point


def unpack_int4(q_packed: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """
    (for testing) map packed INT4 back to fp32 tensor
    """
    shape = q_packed.shape[:-1] + (-1,)          # [-1] already N//2 for 4-bit
    qw = q_packed.view(-1)
    out = torch.zeros(qw.numel() * 8, dtype=torch.float32, device=qw.device)
    for shift in range(8):
        unpacked = (qw >> (shift * 4)) & 0xF
        out[shift::8] = unpacked.to(torch.float32)
    out = (out - zero_point.view(-1)) * scale.view(-1)
    out = out.view(shape)
    return out


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
