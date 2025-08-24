
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256,'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128,'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128,'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,num_warps=8),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['c_ptr']
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
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

    offs_m  = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n  = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k  = pid_z * (BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk) + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_off = k * BLOCK_SIZE_K * SPLIT_K
        k_now = k_off + offs_k
        a = tl.load(a_ptrs, mask=k_now[None, :] < K, other=0.0)

        b_packed = tl.load(b_ptrs, mask=k_now[:, None] < K, other=0)
        b_shift  = ((k_now[:, None] % 8) * 4)
        b_i4     = (b_packed >> b_shift) & 0xF

        g_id  = k_now[:, None] // group_size
        b_scale = tl.load(bs_ptr + g_id * stride_bsk + offs_n[None, :] * stride_bsn,
                          mask=k_now[:, None] < K, other=0.0)
        b_zero  = tl.load(bzp_ptr + g_id * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn,
                          mask=k_now[:, None] < K, other=0)

        zp_shift = ((offs_n[None, :] % 8) * 4)
        b_zp_i4  = (b_zero >> zp_shift) & 0xF

        b_float = (b_i4 - b_zp_i4) * b_scale
        acc += tl.dot(a, b_float.to(a.dtype))

        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K // 8) * stride_bk

    c = acc
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs  = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask    = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=mask)


def matmul_dequantize_int4_s2(x: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor,
                              qzeros: torch.IntTensor, group_size: int = 128, output=None) -> torch.FloatTensor:
    M, K = x.shape
    _, N = scales.shape
    assert K == qweight.shape[0] * 8
    if output is None:
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
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
        group_size, )
    return output


def quantize_int4(w: torch.Tensor, group_size: int = 128):
    w = w.float()
    oc, ic = w.shape          # (K, N) un-transposed
    assert ic % group_size == 0
    w = w.view(oc, ic // group_size, group_size)

    wmax = w.max(dim=2, keepdim=True)[0]
    wmin = w.min(dim=2, keepdim=True)[0]
    scale = (wmax - wmin) / 15.0
    zero  = torch.round(-wmin / scale).clamp(0, 15).to(torch.uint8)

    q = torch.round((w - wmin) / scale).clamp(0, 15).to(torch.uint8)

    q = q.view(oc, ic)
    zero = zero.view(oc, ic // group_size)

    ncols = ic
    packed = torch.zeros((oc, ncols // 8), dtype=torch.int32, device=w.device)
    for i in range(0, ncols, 8):
        chunk = q[:, i:i+8].to(torch.int32)
        packed[:, i//8] = (
            chunk[:,7] << 28 |
            chunk[:,6] << 24 |
            chunk[:,5] << 20 |
            chunk[:,4] << 16 |
            chunk[:,3] << 12 |
            chunk[:,2] << 8  |
            chunk[:,1] << 4  |
            chunk[:,0]
        )

    zero_packed = torch.zeros((oc, (ncols // group_size + 7) // 8),
                              dtype=torch.int32, device=w.device)
    nz = zero.shape[1]
    for i in range(0, nz, 8):
        zchunk = zero[:, i:i+8].to(torch.int32)
        idx = torch.arange(zchunk.size(1), device=w.device)
        zpacked = torch.sum(zchunk << (idx * 4), dim=1, keepdim=True)
        zero_packed[:, i//8] = zpacked.squeeze(1)

    return packed.view(torch.int32), scale.squeeze(-1).half(), zero_packed.view(torch.int32)


def unpack_int4(w_packed: torch.IntTensor, scale: torch.FloatTensor,
                zero: torch.IntTensor, group_size: int = 128):
    oc, Nw = w_packed.shape
    ic = Nw * 8
    num_groups = ic // group_size
    scale = scale.view(oc, num_groups, 1).expand(-1, -1, group_size).reshape(oc, ic)
    zero_shape = (oc, num_groups)
    nz = zero.shape[1] * 8
    zero = zero.view(oc, nz)[:, :num_groups]
    zero = zero.view(oc, num_groups, 1).expand(-1, -1, group_size).reshape(oc, ic)

    w_bytes = w_packed.view(torch.uint8).view(oc, ic // 2)
    w = torch.empty((oc, ic), dtype=torch.uint8, device=w_packed.device)
    for k in range(0, ic, 2):
        b = w_bytes[:, k//2]
        w[:, k]   = (b & 0xF).to(torch.uint8)
        w[:, k+1] = ((b >> 4) & 0xF).to(torch.uint8)

    return (w.float() - zero.float()) * scale.float()


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
