
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1
        }, num_stages=2, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1
        }, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
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
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    BLOCK_K_S = BLOCK_SIZE_K * SPLIT_K
    offs_k = pid_k * BLOCK_K_S + tl.arange(0, BLOCK_K_S)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K_S)):
        k_offs = k * BLOCK_K_S + offs_k[None, :]
        a_mask = (offs_am[:, None] < M) & (k_offs < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0)
        g_idx = ( offs_k[:, None] // group_size )
        scales = tl.load(scales_ptr + g_idx * stride_scales_g + offs_bn[None, :] * stride_scales_n)
        zeros  = tl.load(zeros_ptr  + g_idx * stride_zeros_g  + (offs_bn[None, :] // 8) * stride_zeros_n)
        shift  = (offs_k[:, None] % 8) * 4
        zp_shift = (offs_bn[None, :] % 8) * 4
        b_vals = (b >> shift) & 0xF
        b_zp   = (zeros >> zp_shift) & 0xF
        b_fp = (b_vals - b_zp) * scales
        acc += tl.dot(a.to(tl.float16), b_fp.to(tl.float16))
        a_ptrs += BLOCK_K_S * stride_ak
        b_ptrs += (BLOCK_K_S // 8) * stride_bk
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=mask)


def quantize_int4(w: torch.Tensor, group_size: int = 128):
    assert w.dim() == 2
    K_, N = w.shape
    assert K_ % group_size == 0
    K = K_
    w = w.view(-1, group_size, N)
    wmin = torch.amin(w, dim=1, keepdim=True)
    wmax = torch.amax(w, dim=1, keepdim=True)
    scale = (wmax - wmin) / 15.
    zero  = (-wmin / scale).round().clamp(0, 15).to(torch.int32)
    q = (w / scale + zero).round().clamp(0, 15).to(torch.int32)
    q = q.to(torch.uint8)
    packed = (q[::2, :, :] | (q[1::2, :, :] << 4)).view(-1, N)
    scales = scale.squeeze(1).contiguous()
    zeros  = zero.squeeze(1).contiguous()
    return packed, scales, zeros, None


def matmul_dequantize_int4_s2(
    x: torch.FloatTensor, qweight: torch.IntTensor,
    scales: torch.FloatTensor, qzeros: torch.IntTensor,
    group_size: int = 128, output=None
) -> torch.FloatTensor:
    assert x.is_contiguous()
    M, K = x.shape
    Kw, N = qweight.shape
    assert K == Kw * 8
    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=torch.float16)
    else:
        output.fill_(0.0)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K']
    )
    matmul_kernel[grid](
        x, qweight, output,
        scales, qzeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size
    )
    return output


def unpack_int4(b_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    b_packed = b_packed.contiguous()
    Kw, N = b_packed.shape
    K = Kw * 8
    b_uint = torch.zeros((K, N), dtype=torch.uint8, device=b_packed.device)
    b_uint[0::2, :] = b_packed & 0xF
    b_uint[1::2, :] = (b_packed >> 4) & 0xF
    group_idx = torch.arange(K, device=b_packed.device) // group_size
    scl = scales[group_idx, :]
    z   = zeros[group_idx, :]  # shape (K, N)
    z = ((z.view(-1, 1, N) >> (4 * torch.arange(N//8, device=b_packed.device)[None, :, None])) & 0xF).view(z.shape[0], -1)[:, :N]
    b_fp = (b_uint.to(torch.float32) - z) * scl
    return b_fp


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
