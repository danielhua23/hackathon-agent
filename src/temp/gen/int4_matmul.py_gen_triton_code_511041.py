
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    bs_ptr, bzp_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bsk, stride_bsn,
    stride_bzpk, stride_bzpn,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_start = pid_z * (BLOCK_SIZE_K * SPLIT_K) + tl.arange(0, BLOCK_SIZE_K * SPLIT_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_start[None, :] * stride_ak
    b_ptrs = b_ptr + ((offs_k_start[:, None] // 8) * stride_bk) + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=(offs_k_start[None, :] < K), other=0.0)
        b_i32 = tl.load(b_ptrs, mask=(offs_k_start[:, None] < K), other=0)

        n_idx = offs_n[None, :]
        k_idx = offs_k_start[:, None]
        mask_valid = (k_idx < K)

        group_id_k = k_idx // group_size
        scales = tl.load(bs_ptr + group_id_k * stride_bsk + n_idx * stride_bsn, mask=mask_valid, other=0.0)
        zeros = tl.load(bzp_ptr + group_id_k * stride_bzpk + (n_idx // 8) * stride_bzpn, mask=mask_valid, other=0)

        b_shift = ((k_idx % 8) * 4)
        zp_shift = ((n_idx % 8) * 4)

        b_i4 = (b_i32 >> b_shift) & 0xF
        zp_i4 = (zeros >> zp_shift) & 0xF
        b_float = (b_i4 - zp_i4).to(tl.float32) * scales.to(tl.float32)

        accumulator += tl.dot(a.to(tl.float32), b_float.to(tl.float32))

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk

    c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=mask)


def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, group_size: int = 128, output=None) -> torch.FloatTensor:
    assert x.ndim == 2 and qweight.ndim == 2
    assert x.shape[-1] == (qweight.shape[0] * 8)
    assert x.is_contiguous()

    M, K = x.shape
    N = scales.shape[1]
    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid_fn(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META['SPLIT_K'])

    matmul_kernel[grid_fn](
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


def quantize_int4(w: torch.Tensor, group_size: int = 128):
    assert w.dim() == 2
    w = w.float()
    oc, ic = w.shape
    assert ic % group_size == 0
    w = w.reshape(oc, ic // group_size, group_size)

    wmax = w.amax(dim=2, keepdim=True)
    wmin = w.amin(dim=2, keepdim=True)
    scale = (wmax - wmin) / 15.0
    zero = (-wmin / scale).round().clamp(0, 15).to(torch.int8)

    int_w = ((w - wmin) / scale).round().clamp(0, 15).to(torch.int8)

    int_w_reshaped = int_w.view(oc, ic)
    zero_reshaped = zero.view(oc, -1)

    col_bytes = torch.empty(oc, ic // 2, dtype=torch.int8, device=w.device)
    for j in range(0, ic, 2):
        lo = int_w_reshaped[:, j]
        hi = int_w_reshaped[:, j + 1]
        packed = (hi << 4) | lo
        col_bytes[:, j // 2] = packed.to(torch.int8)

    out = col_bytes.view(oc, ic // 8).view(torch.int32)
    return out, scale.squeeze(-1).half(), zero_reshaped


def unpack_int4(w_packed: torch.IntTensor, scale: torch.Tensor, zero: torch.Tensor, group_size: int = 128):
    oc, ic_bytes = w_packed.shape
    ic = ic_bytes * 8
    assert ic % group_size == 0

    w_int = torch.empty(oc, ic, dtype=torch.int8, device=w_packed.device)
    packed = w_packed.view(torch.int8).view(oc, ic // 8)
    for j in range(ic // 8):
        b = packed[:, j]
        for k in range(8):
            val = (b >> (k * 4)) & 0xF
            w_int[:, j * 8 + k] = val

    num_groups = ic // group_size
    scale = scale.view(oc, num_groups, 1).expand(-1, -1, group_size).reshape(oc, ic)
    zero = zero.view(oc, num_groups, 1).expand(-1, -1, group_size).reshape(oc, ic)
    return (w_int.float() - zero.float()) * scale.float()


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
