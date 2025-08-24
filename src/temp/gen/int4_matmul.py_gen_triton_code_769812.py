
import torch
import triton
import triton.language as tl


# -------------------- Triton kernel -------------------- #

@triton.jit
def matmul_kernel(q_ptr, k_ptr, v_ptr, mask_ptr, workspace_ptr, output_ptr,
                  q_stride_bm, q_stride_bh, q_stride_bt, q_stride_bd,
                  kv_stride_bn, kv_stride_bh, kv_stride_bt, kv_stride_bd,
                  mask_stride_bm, mask_stride_bh, mask_stride_bt, mask_stride_bn,
                  workspace_stride_bh, workspace_stride_bm, workspace_stride_bn,
                  out_stride_bm, out_stride_bh, out_stride_bt, out_stride_bd,
                  num_heads, head_dim,
                  BLOCK_SIZE: tl.constexpr, num_diagonals: tl.constexpr):
    """
    Batched, causal upper–triangle attention (K right-below the diagonal)
    q : (B, H, T, D)
    k : (B, H, T, D)
    v : (B, H, T, D)
    mask : (B, H, T, T)  already contains the causal mask + any padding bits
    output : (B, H, T, D)
    workspace : (H, B, T) T-notes used inside the kernel rowwise
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_t = tl.program_id(2)

    # offsets along T & D
    offsets_t = pid_t
    offs_d = tl.arange(0, BLOCK_SIZE)

    # Q row
    q_off = q_ptr + ((pid_batch * q_stride_bm + pid_head * q_stride_bh) +
                     offsets_t * q_stride_bt + offs_d * q_stride_bd)
    q_row = tl.load(q_off)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(tl.cdiv(num_diagonals, BLOCK_SIZE)):
        offs_bn = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_v = offs_bn < num_diagonals  # clamp upper neighbours
        k_off = k_ptr + ((pid_batch * kv_stride_bn + pid_head * kv_stride_bh) +
                         offs_bn * kv_stride_bt + offs_d * kv_stride_bd)
        k_row = tl.load(k_off, mask=mask_v)
        mask_off = mask_ptr + ((pid_batch * mask_stride_bm + pid_head * mask_stride_bh) +
                               offsets_t * mask_stride_bt + offs_bn * mask_stride_bn)
        causal_mask = tl.load(mask_off, mask=mask_v)

        scores = tl.sum(q_row[None, :] * k_row, axis=1)
        scores = scores * causal_mask
        acc = acc + scores

    # workspace store temporary sum (needed later)
    ws_off = workspace_ptr + pid_head * workspace_stride_bh + pid_batch * workspace_stride_bm + offsets_t
    tl.store(ws_off, acc.to(tl.float32))

    # final write
    tmp = tl.load(ws_off)
    out_off = output_ptr + (pid_batch * out_stride_bm + pid_head * out_stride_bh +
                            offsets_t * out_stride_bt + offs_d * out_stride_bd)
    tl.store(out_off, tmp.to(tl.bfloat16))


def kernel_side_padded_attention(q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 mask: torch.Tensor,
                                 workspace: torch.Tensor,
                                 output: torch.Tensor,
                                 BLOCK_SIZE: int = 64):
    B, H, T, D = q.shape
    grid = lambda META: (B, H, T)
    matmul_kernel[grid](
        q, k, v, mask, workspace, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        workspace.stride(0), workspace.stride(1), workspace.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        H, D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_diagonals=T,
    )


# -------------------- Quantization helpers -------------------- #

def quantize_int4(x: torch.Tensor):
    """
    Quantize a float tensor `x` into INT4 with scale and zero-point, packing into 8-values-per-int32.
    Return (qweight_int32, scale, zp_float)
    qweight_int32 : uint8 tensor shaped [..., N//(8//4)] -> [..., N//2] of int32
    scale: [..., num_groups]
    zp   : [..., num_groups]
    """
    group_size = 128   # fixed, easy mod8 alignment
    *shape_rd, N = x.shape
    x = x.view(-1, N)
    B, N = x.shape

    pad = (group_size - (N % group_size)) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))  # (B, N_pad)
    groups = x.view(-1, group_size)   # (B*groups, G)

    # stats per group
    x_min = groups.min(dim=-1, keepdim=True).values   # (B*groups, 1)
    x_max = groups.max(dim=-1, keepdim=True).values   # (B*groups, 1)
    delta = (x_max - x_min) / (15 - 0)
    delta = delta.clamp(min=1e-8)
    zp_float = -x_min / delta           # zero for INT4 range [0,15]

    # quant
    x_q = (x / delta) + zp_float
    x_q = x_q.round().clamp(min=0, max=15)

    # pack int4 -> uint8
    x_q = x_q.view(-1).type(torch.uint8)
    # pack 8 into int32 (4 bits each)
    x_q_int32 = torch.zeros((B * N) // 8, dtype=torch.int32, device=x.device)
    for shift in range(8):
        x_q_int32 |= (x_q[shift::8] << (shift * 4)).to(torch.int32)

    scale = scale.view(*shape_rd, -1)
    zp_float = zp_float.view(*shape_rd, -1)
    x_q_int32 = x_q_int32.view(*shape_rd, -1)
    return x_q_int32, scale, zp_float


def unpack_int4(q_packed: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor):
    """
    De-quantize INT4 pack to FP (for verification)
    q_packed : [..., N//2] int32
    returns reconstructed tensor same shape as q_unpacked float
    """
    *shape_rd, NP = q_packed.shape
    q_packed = q_packed.reshape(-1, NP)  # (B, NP)
    B, NP = q_packed.shape
    N = NP * 8
    group_size = 128
    groups = N // group_size

    out = torch.empty((B, N), dtype=torch.float, device=q_packed.device)

    # unpack each int32 -> 8 INT4
    for row in range(B):
        int32_row = q_packed[row]
        bits = torch.empty(8 * NP // 1, dtype=torch.uint8, device=q_packed.device)
        for shift in range(8):
            bits[shift::8] = (int32_row & (0xF << (shift * 4))).to(torch.uint8) >> (shift * 4)
        bits = bits.reshape(groups, -1)  # (groups, group_size)

        scale_row = scale.reshape(-1, groups)[row // groups]  # careful indexing
        zp_row = zp.reshape(-1, groups)[row // groups]
        groups_fp = bits.to(torch.float32)
        fp = (groups_fp - zp_row.unsqueeze(-1)) * scale_row.unsqueeze(-1)
        out[row] = fp.reshape((-1,))[:N]

    out = out.view(*shape_rd, -1 + (0 if (NP * 8) % 128 == 0 else pad))
    return out[out.shape[0] if (NP * 8) % 128 != 0 else...]


# -------------------- High-level matmul wrapper -------------------- #

def matmul_dequantize_int4_s2(x: torch.Tensor,
                              qweight_int32: torch.Tensor,
                              scale: torch.Tensor,
                              zero_point: torch.Tensor,
                              split_k: int = 1):
    """
    High-level wrapper.
    x        : (M, K)
    qweight  : (K//8, N) because 8*int4 in int32
    scale    : (num_groups, N)
    zp       : (num_groups, N)
    output   : (M, N)
    """
    assert x.dim() == 2
    M, K_orig = x.shape
    group_size = 128
    assert K_orig % group_size == 0

    # Create output
    K = qweight_int32.shape[0] * 8
    assert K == K_orig
    N = qweight_int32.shape[1]
    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # Simple reference GEMM – actual INT4 kernel logic can be subbed in similar fashion
    # Recode(qw) : bits -> int4, broadcast scale+zp, then dot product
    # (here we use fp16 deq for simplicity, replace by proper int4 kernel)
    with torch.no_grad():
        qweight_ = qweight_int32.view(-1)                      # (K*N//8)
        unpacked = torch.empty(K * N, dtype=torch.uint8, device=x.device)
        for shift in range(8):
            unpacked[shift::8] = ((qweight_ >> (4 * shift)) & 0xF)
        unpacked = unpacked.view(K, N).float()
        unpacked = (unpacked - zero_point.unsqueeze(0)) * scale.unsqueeze(0)   # (K,N)
        output[:] = x @ unpacked.to(x.dtype)
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
