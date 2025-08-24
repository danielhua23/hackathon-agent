
import torch
import triton
import triton.language as tl

configs_matmul = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N':  32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
    triton.Config({'BLOCK_SIZE_M':  64, 'BLOCK_SIZE_N':  64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
]

@triton.autotune(
    configs=configs_matmul,
    key=["M", "N", "K"],
    use_cuda_graph=False
)
@triton.jit
def matmul_kernel(
    A, B, C, scales, zeros,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g0, stride_zeros_n,
    groupsize,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    GROUP_SIZE_M: tl.constexpr = 8
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)

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
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + ((offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_pos = k * BLOCK_SIZE_K * SPLIT_K + offs_k
        g_idx = (k_pos) // groupsize

        mask_k = k_pos < K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)

        offset_b = (k_pos[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
        b_chunk = tl.load(B + offset_b, mask=mask_k[:, None], other=0)

        scale_offset = g_idx[:, None] * stride_scales_g + offs_n[None, :] * stride_scales_n
        scale_val = tl.load(scales + scale_offset, mask=mask_k[:, None], other=0.0)

        zp_val = tl.load(zeros + g_idx[:, None] * stride_zeros_g0 + (offs_n // 8)[None, :] * stride_zeros_n, mask=mask_k[:, None], other=0.0)
        shift_n = (offs_n % 8)[None, :] * 4
        inv_zp = ((zp_val >> shift_n) & 0xF) * scale_val

        shift_k = (k_pos % 8)[:, None] * 4
        w_int = (b_chunk >> shift_k) & 0xF
        w_fp = (w_int * scale_val - inv_zp)

        accumulator += tl.dot(a, w_fp)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak

    c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cm = offs_cm < M
    mask_cn = offs_cn < N
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = mask_cm[:, None] & mask_cn[None, :]

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=mask)

def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int = 128, output=None) -> torch.FloatTensor:
    assert x.is_contiguous(), "A must be contiguous"
    Kx, N = qweight.shape
    K = Kx * 8
    M = x.shape[0]
    assert x.shape[1] == K, f"A second dim {x.shape[1]} must equal weight rows {K}"
    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    else:
        assert output.shape == (M, N), "output shape must be (M, N)"
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        max(META.get('SPLIT_K', 1), 1),
    )
    num_groups = max(1, K // group_size)
    second_dim = 1 if N <= 8 else (N + 7) // 8
    matmul_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size,
    )
    return output

configs_dequant = [
    triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 128}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=4),
]

@triton.autotune(
    configs=configs_dequant,
    key=["K", "N"],
    use_cuda_graph=False
)
@triton.jit
def dequantize_kernel(
    qw_ptr, sc_ptr, zp_ptr, fpw_ptr,
    K, N, group_size,
    stride_qk, stride_qn,
    stride_scg, stride_scn,
    stride_zpg, stride_zpn,
    stride_fk, stride_fn,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    k_blk = tl.program_id(0)
    n_blk = tl.program_id(1)

    offs_k = k_blk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = n_blk * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_k = offs_k[:, None] < K
    mask_n = offs_n[None, :] < N
    mask = mask_k & mask_n

    grp = offs_k[:, None] // group_size

    qw_offs = (offs_k[:, None] // 8) * stride_qk + offs_n[None, :] * stride_qn
    qw_local = tl.load(qw_ptr + qw_offs, mask=mask, other=0)

    sc_offs = grp * stride_scg + offs_n[None, :] * stride_scn
    sc_local = tl.load(sc_ptr + sc_offs, mask=mask, other=0.0)

    zp_offs = grp * stride_zpg + (offs_n // 8)[None, :] * stride_zpn
    zp_quad = tl.load(zp_ptr + zp_offs, mask=mask, other=0)

    bits = 4
    shift_k = (offs_k % 8)[:, None] * bits
    shift_n = (offs_n % 8)[None, :] * bits

    qh = (qw_local >> shift_k) & 0xF
    qz = (zp_quad >> shift_n) & 0xF

    dq_val = (qh - qz) * sc_local
    tl.store(fpw_ptr + offs_k[:, None] * stride_fk + offs_n[None, :] * stride_fn, dq_val, mask=mask)

def quantize_int4(x: torch.Tensor, groupsize: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    W = x.t().contiguous()
    K_raw, N = W.shape
    assert K_raw % groupsize == 0, "K must be divisible by groupsize"
    groups = K_raw // groupsize
    W = W.view(groups, groupsize, N)
    wmin = W.min(dim=1, keepdim=True)[0]
    wmax = W.max(dim=1, keepdim=True)[0]
    scale = (wmax - wmin) / 15
    zero = -wmin / scale
    zero = torch.round(zero).clamp(0, 15)

    qweight = torch.clamp(torch.round(W / scale + zero), 0, 15)
    qweight = qweight.to(torch.int8)

    packed = torch.zeros((groups * groupsize) // 8, N, dtype=torch.int32, device=x.device)
    for col in range(N):
        w_col = qweight[:, :, col].flatten()
        for idx in range(0, w_col.size(0), 8):
            v = w_col[idx:idx+8].view(torch.uint8)
            v &= 0xF
            v1 = v[::2] | (v[1::2] << 4)
            packed[idx//8, col] = v1.view(torch.int32)
    packed = packed.t()
    qweight = packed

    scale = scale.squeeze(1).view(groups, N).t()
    zero = zero.squeeze(1).view(groups, N).t()

    qzeros = torch.zeros_like(zero, dtype=torch.int32)
    for col in range(N):
        z_col = zero[:, col]
        for idx in range(0, groups, 8):
            seg = z_col[idx: idx+8]
            tmp = 0
            for k, val in enumerate(seg):
                tmp |= (val.int() & 0xF) << (4 * k)
            qzeros[idx//8, col] = tmp

    return qweight.contiguous(), scale.contiguous(), qzeros.contiguous()

def unpack_int4(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    groupsize: int
) -> torch.Tensor:
    K_pack, N = packed_weights.shape
    K = K_pack * 8
    device = packed_weights.device
    dtype = scales.dtype

    unpacked = torch.zeros((K, N), dtype=dtype, device=device)
    for i in range(K_pack):
        chunk = packed_weights[i, :]
        for j in range(8):
            bits = (chunk >> (j * 4)) & 0xF
            unpacked[i * 8 + j, :] = bits.to(dtype)

    num_groups = K // groupsize
    scale_ex = scales.t().repeat_interleave(groupsize, dim=0)
    zeros_ex = zeros.t().repeat_interleave(groupsize, dim=0)

    zeros_unp = torch.zeros((K, N), dtype=dtype, device=device)
    for i in range(num_groups):
        z_chunk = zeros.t()[i, :]
        for j in range(groupsize):
            for n in range(N):
                g_group = i * groupsize + j
                zeros_unp[g_group, n] = z_chunk[n]

    result = (unpacked - zeros_unp) * scale_ex
    return result.t()


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
