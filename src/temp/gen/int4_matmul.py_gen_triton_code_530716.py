
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------
# Triton kernel: matmul with on-the-fly INT4 de-quantisation
# ------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

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
    offs_k  = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_off = k * BLOCK_SIZE_K * SPLIT_K
        mask_k = (offs_k[None, :] + k_off) < K
        mask_a = (offs_am[:, None] < M) & mask_k
        mask_b = mask_k & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs + k_off * stride_ak, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs + (k_off // 8) * stride_bk, mask=mask_b, other=0.0)

        group_idx = (offs_k[None, :] + k_off) // group_size
        bs   = tl.load(bs_ptr   + group_idx * stride_bsk   + offs_bn[None, :] * stride_bsn,   mask=mask_b, other=0.0)
        bzps = tl.load(bzp_ptr  + group_idx * stride_bzpk  + (offs_bn[None, :] // 8) * stride_bzpn, mask=mask_b, other=0.0)

        b_shift = ((offs_k[None, :] + k_off) % 8) * 4
        bzp_shift = (offs_bn[None, :] % 8) * 4

        int4_b   = (b    >> b_shift)   & 0xF
        int4_bzp = (bzps >> bzp_shift) & 0xF

        b_deq = ((int4_b - int4_bzp) * bs).to(tl.float16)
        accumulator += tl.dot(a.to(tl.float16), b_deq)

    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn
    mask   = ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] < M) & ((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < N)
    if SPLIT_K > 1:
        tl.atomic_add(c_ptrs, c, mask=mask)
    else:
        tl.store(c_ptrs, c, mask=mask)

# ------------------------------------------------------------
# Wrapper: launch the matmul kernel
# ------------------------------------------------------------
def matmul_dequantize_int4_s2(x: torch.Tensor, qweight: torch.Tensor,
                              scales: torch.Tensor, zeros: torch.Tensor,
                              group_size: int = 128) -> torch.Tensor:
    assert x.is_contiguous()
    assert qweight.is_contiguous()
    assert scales.is_contiguous()
    assert zeros.is_contiguous()

    M, K = x.shape
    N = scales.shape[1]

    output = torch.empty((M, N), device=x.device, dtype=torch.float16)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
                META['SPLIT_K'])

    matmul_kernel[grid](
        x, qweight, output,
        scales, zeros,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        group_size,
        GROUP_SIZE_M=8, SPLIT_K=1
    )
    return output


# ------------------------------------------------------------
# Triton kernel: INT4 quantisation (packing helper)
# ------------------------------------------------------------
@triton.jit
def pack_kernel(
    src_ptr, dst_ptr, scales_ptr, zeros_ptr,
    stride_sr, stride_sc,
    stride_dr, stride_dc,
    stride_s, stride_z,
    BLOCK_M: tl.constexpr,  # rows handled (tile)
    BLOCK_N: tl.constexpr,  # cols handled (tile)
    GROUP_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    gs  = tl.program_id(1)

    col_start = gs * GROUP_SIZE
    col_off   = tl.arange(0, BLOCK_N)
    cols = col_start + col_off

    mask = cols < stride_sc  # valid in the row
    vals = tl.load(src_ptr + row * stride_sr + cols, mask=mask, other=0.0)

    max_val = tl.max(vals, axis=0)
    min_val = tl.min(vals, axis=0)
    scale = (max_val - min_val) / 15.0
    zero  = -min_val / scale

    s_idx = row * (stride_sc // GROUP_SIZE) + gs
    tl.store(scales_ptr + s_idx, scale.to(tl.float16))
    tl.store(zeros_ptr  + s_idx,  zero.to(tl.float16))

    for shift in range(0, GROUP_SIZE, 8):
        # 8 contiguous floats
        idx = shift + tl.arange(0, 8)
        msk = (col_start + idx) < stride_sc
        v   = tl.load(src_ptr + row * stride_sr + col_start + idx, mask=msk, other=0.0)

        q   = ((v / scale + zero) + 0.5).to(tl.int32)
        q   = tl.maximum(tl.minimum(q, 15), 0)

        packed = tl.full([8], 0, dtype=tl.int32)
        for ch in range(8):
            packed = tl.where(msk,
                              packed | (q[ch] << (ch * 4)),
                              packed)

        col_int = (col_start + shift) // 8
        tl.store(dst_ptr + row * stride_dr + col_int, packed[0])


# ------------------------------------------------------------
# Wrapper: quantise a weight matrix down to INT4
# ------------------------------------------------------------
def quantize_int4(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weight.dim() == 2
    rows, cols = weight.shape

    packed = torch.empty((rows, cols // 8), dtype=torch.int32, device=weight.device)
    scales = torch.empty((rows, cols // group_size), dtype=torch.float16, device=weight.device)
    zeros  = torch.empty_like(scales)

    grid = lambda _: (rows, cols // group_size)

    pack_kernel[grid](
        weight, packed, scales, zeros,
        weight.stride(0), weight.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0), zeros.stride(0),
        BLOCK_M=1, BLOCK_N=group_size, GROUP_SIZE=group_size
    )
    return packed, scales, zeros


# ------------------------------------------------------------
# Helper: unpack INT4 back to float16 – use only for testing
# ------------------------------------------------------------
def unpack_int4(qw_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                group_size: int = 128) -> torch.Tensor:
    rows, packs = qw_packed.shape
    cols = packs * 8
    weight = torch.empty((rows, cols), dtype=torch.float16, device=qw_packed.device)

    for r in range(rows):
        for g in range(scales.shape[1]):
            s = scales[r, g].item()
            z = zeros[r, g].item()
            start = g * group_size
            end   = min(start + group_size, cols)
            pack_off = start // 8
            num_packs = (end - start + 7) // 8
            for pw in range(num_packs):
                p = qw_packed[r, pack_off + pw].item()
                for k in range(8):
                    val = (p >> (k * 4)) & 0xF
                    pos = start + pw * 8 + k
                    if pos < cols:
                        weight[r, pos] = val * s + z
    return weight


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
