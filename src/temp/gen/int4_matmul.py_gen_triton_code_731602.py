
import torch
import triton
import triton.language as tl
import math


@triton.jit
def matmul_kernel(
    A, B, C, scales, zeros,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_scales_g, stride_scales_n,
    stride_zeros_g, stride_zeros_n,
    M, N, K, group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_k = tl.program_id(axis=1)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    A = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        k_idx = k * BLOCK_SIZE_K * SPLIT_K + offs_k
        if k_idx[0] >= K:
            break

        a = tl.load(A)
        b_raw = tl.load(B)  # int32 weights

        # Dequantize INT4
        g_id = (k * BLOCK_SIZE_K) // group_size
        scales_val = tl.load(scales + g_id * stride_scales_g + offs_n * stride_scales_n)
        zeros_val = tl.load(zeros + g_id * stride_zeros_g + offs_n * stride_zeros_n)

        # Unpack INT4
        b_i4 = ((b_raw >> (4 * (offs_k[:, None] % 8))) & 0xF).to(tl.float32)
        b = (b_i4 - zeros_val[None, :]) * scales_val[None, :]

        acc += tl.dot(a, b)

        A += BLOCK_SIZE_K * SPLIT_K * stride_ak
        B += BLOCK_SIZE_K * SPLIT_K * stride_bk

    if SPLIT_K > 1:
        C_offs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.atomic_add(C_offs, acc.to(C.dtype.element_ty))
    else:
        C = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(C, acc.to(C.dtype.element_ty))


def matmul_dequantize_int4_s2(
    x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
    zeros: torch.Tensor, group_size: int, output: torch.Tensor = None,
):
    M, K = x.shape
    N, K_p = qweight.shape
    assert K_p == K // 8, "Weight matrix K dimension mismatch (packed)"
    assert K % 8 == 0, "K must be divisible by 8 for INT4 packing"
    K_padded = triton.next_power_of_2(K)

    if output is None:
        output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    SPLIT_K = 1

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            META['SPLIT_K'],
        )

    matmul_kernel[grid](
        x, qweight, output, scales, zeros,
        x.stride(0), x.stride(1),
        qweight.stride(1), qweight.stride(0),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        M, N, K, group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M, SPLIT_K=SPLIT_K,
    )

    return output


def quantize_int4(x: torch.Tensor, group_size: int = 128):
    x = x.t().contiguous().cuda()
    N, K = x.shape
    assert K % group_size == 0, f"Weight columns ({K}) must be divisible by group_size ({group_size})"

    x = x.view(N, K // group_size, group_size)
    x_f = x.to(torch.float32)
    x_min = x_f.min(dim=-1, keepdim=True)[0]
    x_max = x_f.max(dim=-1, keepdim=True)[0]
    
    scales = (x_max - x_min) / 15.0
    zeros = (-x_min / scales).round().clamp(0, 15)
    x_q = (x_f / scales + zeros).round().clamp(0, 15)

    scales = scales.squeeze(-1)
    zeros = zeros.squeeze(-1)

    # Pack INT4 to INT32
    x_q = x_q.view(N, K // 8)  # Each 8 INT4s packed
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=x.device)
    for i in range(8):
        packed |= (x_q[:, i::8].to(torch.int32) << (4 * i))

    return packed.t().contiguous(), scales.t().contiguous(), zeros.t().contiguous()


def unpack_int4(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int = 128):
    qweight = qweight.cuda()
    scales = scales.cuda()
    zeros = zeros.cuda()

    N, K_p = qweight.shape
    K = K_p * 8

    weights = torch.zeros((N, K), dtype=torch.float16, device=qweight.device)
    group_num = K // group_size

    for i in range(8):
        bits = (qweight >> (4 * i)) & 0xF
        idx = torch.arange(i, K, 8, device=qweight.device)
        weights[:, idx] = (bits - zeros[:, idx // group_size]) * scales[:, idx // group_size]

    return weights


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
