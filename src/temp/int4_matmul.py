
import torch
import triton
import triton.language as tl

# ========== Quantization utilities ==========

def quantize_int4(tensor: torch.Tensor, group_size: int = 128):
    """
    Quantize 2-D (K, N) float tensor to INT4 with GPTQ layout.
    Returns:
        qweight    int32 [K//8, N]
        scales     float [K//group_size, N]
        qzeros     int32 [K//group_size, N//8]   (packed 4-bit)
        group_size int (unchanged)
    """
    assert tensor.dim() == 2, "expecting 2-D tensor"
    K, N = tensor.shape
    device = tensor.device

    groups = (K + group_size - 1) // group_size
    scales = torch.empty((groups, N), device=device, dtype=torch.float32)
    qzeros = torch.empty((groups, N // 8), device=device, dtype=torch.int32)
    qweight = torch.empty((K // 8, N), device=device, dtype=torch.int32)

    ones = (2 ** (4 * torch.arange(8, device=device, dtype=torch.int32))).view(1, 8)

    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, K)
        group_len = end - start
        gblk = tensor[start:end, :].float()

        gmin = gblk.min(0)[0]
        gmax = gblk.max(0)[0]
        scale = (gmax - gmin) / 15.0
        zero = (-gmin / scale).round().clamp(0, 15).to(torch.int32)

        zero_col = zero.view(-1, 8)
        packed_zero = (zero_col * ones).sum(dim=1, dtype=torch.int32)
        qzeros[g] = packed_zero

        q = torch.round(gblk / scale + zero.view(1, N)).clamp(0, 15).to(torch.int32)
        q_rows = q.view(-1, 8, N) * ones.view(1, 8, 1)
        q_packed = q_rows.sum(dim=1, dtype=torch.int32)
        qweight[start // 8 : end // 8] = q_packed

        scales[g] = scale.to(tensor.dtype)

    scales = scales.to(tensor.dtype)
    return qweight, scales, qzeros, group_size


def unpack_int4(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
                group_size: int = 128, thread_local: bool = False) -> torch.Tensor:
    Kdiv8, N = qweight.shape
    K = Kdiv8 * 8
    device = qweight.device
    ones = (2 ** (4 * torch.arange(8, device=device, dtype=torch.int32))).view(1, 8)

    groups = (K + group_size - 1) // group_size
    fp = torch.empty((K, N), device=device, dtype=torch.float32)

    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, K)
        rows = end - start

        packed = qweight[start // 8 : end // 8]
        powers = (2 ** (4 * torch.arange(8, device=device, dtype=torch.int32))).view(-1, 1, 1)
        unpacked = ((packed.unsqueeze(1) >> powers.reshape(1, 8, 1)) & 0xF).view(rows, N)

        zp_packed = qzeros[g]
        zp_unpacked = ((zp_packed.unsqueeze(1) >> powers.reshape(1, 8)).view(-1)[:N] & 0xF)

        sc = scales[g]
        fp[start:end] = (unpacked - zp_unpacked) * sc

    return fp

# ========== Triton kernel (split-K==1 variant)  ==========

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'NO_GROUPS']
)
@triton.jit
def matmul4_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    group_size, NO_GROUPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    BITS = 4
    PACK = 8
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
    offs_k_pack = tl.arange(0, BLOCK_SIZE_K // PACK)
    pack_idx = tl.arange(0, PACK)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (offs_k_pack * PACK + pack_idx)[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k_pack[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_base = k * BLOCK_SIZE_K
        k_actual = k_base + offs_k_pack[:, None] * PACK + pack_idx[None, :]
        mask_k = k_actual[None, :] < K
        mask_m = offs_m[:, None] < M
        mask_a = mask_m & mask_k

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        b_pack = tl.load(b_ptrs)
        b_shift = pack_idx[None, :] * BITS
        b_int = (b_pack >> b_shift) & 0xF

        if NO_GROUPS:
            scale = tl.load(scales_ptr + offs_n * stride_scales_n)
            zero = tl.load(zeros_ptr + (offs_n // 8) * stride_zeros_n)
        else:
            gid = k_base // group_size
            scale = tl.load(scales_ptr + gid * stride_scales_g + offs_n * stride_scales_n)
            zero = tl.load(zeros_ptr + gid * stride_zeros_g + (offs_n // 8) * stride_zeros_n)

        zero_part = (zero >> ((offs_n % 8) * 4)) & 0xF
        b = (b_int - zero_part) * scale
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K // PACK * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_out = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)

# ========== Python wrapper ==========

def matmul_dequantize_int4_s2(
    x: torch.FloatTensor,
    qweight: torch.IntTensor,
    scales: torch.FloatTensor,
    qzeros: torch.IntTensor,
    group_size: int = 128,
    output = None
) -> torch.FloatTensor:
    assert x.is_contiguous(), "x must be contiguous"
    assert qweight.is_contiguous(), "qweight must be contiguous"

    M, K = x.shape
    N = scales.shape[1]
    NO_GROUPS = (group_size == K)

    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    else:
        assert output.shape == (M, N) and output.device == x.device

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    matmul4_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size, NO_GROUPS
    )
    return output

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
