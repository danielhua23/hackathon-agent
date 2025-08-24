
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
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

    BLOCK_K_S = BLOCK_SIZE_K * SPLIT_K
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_sp_k * BLOCK_K_S + tl.arange(0, BLOCK_K_S)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + (offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K_S)):
        k_slice = k * BLOCK_K_S + offs_k[None, :]
        a_mask = (offs_am[:, None] < M) & (k_slice < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_int32 = tl.load(b_ptrs)
        group_idx = k_slice // group_size
        scales = tl.load(scales_ptr + group_idx * stride_scales_g + offs_bn[None, :] * stride_scales_n)
        zeros = tl.load(
            zeros_ptr
            + group_idx * stride_zeros_g
            + (offs_bn[None, :] // 8) * stride_zeros_n
        )

        shift = (k_slice % 8) * 4
        zp_shift = (offs_bn[None, :] % 8) * 4

        b_int4 = (b_int32 >> shift) & 0xF
        b_zp = (zeros >> zp_shift) & 0xF
        b_deq = (b_int4 - b_zp) * scales

        accumulator += tl.dot(a.to(tl.float16), b_deq.to(tl.float16))

        a_ptrs += BLOCK_K_S * stride_ak
        b_ptrs += (BLOCK_K_S // 8) * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask_c)
    else:
        tl.atomic_add(c_ptrs, c, mask=mask_c)

def quantize_int4(w: torch.Tensor, group_size: int = 128):
    w = w.contiguous()
    assert w.dim() == 2
    K, N = w.shape
    assert K % group_size == 0, f"K {K} must be divisible by group_size {group_size}"

    w = w.view(-1, group_size, N)
    wmin = w.amin(dim=1, keepdim=True)
    wmax = w.amax(dim=1, keepdim=True)
    scale = (wmax - wmin) / 15.0
    zero = (-wmin / scale).round().clamp(0, 15)

    wq = ((w / scale + zero).round().clamp(0, 15)).to(torch.int32)

    wq = wq.view(-1, N) # Flatten groups for every row
    packed_w = torch.zeros(K // 8, N, dtype=torch.int32, device=w.device)
    for i in range(8):
        packed_w += (wq[i::8] & 0xF).shl(i * 4).to(torch.int32)

    scale = scale.squeeze(1).contiguous()
    zero = zero.squeeze(1)

    packed_zeros = torch.zeros((K // group_size, N // 8), dtype=torch.int32, device=w.device)
    z_reshaped = zero.view(-1, group_size // 8, 8, N) # [G, group_size//8, 8, N]
    for i in range(8):
        packed_zeros += (z_reshaped[..., i] & 0xF).shl(i * 4).to(torch.int32)

    packed_zeros = packed_zeros.view(-1, N // 8)

    return packed_w.contiguous(), scale.contiguous(), packed_zeros.contiguous(), None

def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int = 128, output=None) -> torch.FloatTensor:
    assert x.is_contiguous(), "Input x must be contiguous"
    assert qweight.is_contiguous(), "Quantized weight must be contiguous"
    M, K = x.shape
    Kw, N = qweight.shape
    assert Kw * 8 == K, f"Unpacked K ({K}) must be 8*Kw. Got Kw={Kw}"

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
        group_size,
    )
    return output

def unpack_int4(b_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    b_packed = b_packed.contiguous()
    Kw, N = b_packed.shape
    K = Kw * 8
    unpacked_b = torch.zeros(K, N, dtype=torch.int32, device=b_packed.device)
    for i in range(8):
        unpacked_b[i::8] = ((b_packed >> (i * 4)) & 0xF).int()

    group_idx = torch.arange(K, device=b_packed.device) // group_size
    s = scales[group_idx, :]
    z = zeros.view(scales.shape[0], -1)[group_idx, :]

    fp = (unpacked_b.float() - z.float()) * s.float()
    return fp


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
