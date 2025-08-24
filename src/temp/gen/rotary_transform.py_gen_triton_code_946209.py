
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLENS,
    stride_x_batch,
    stride_x_head,
    stride_x_m,
    stride_x_n,
    stride_cos_m,
    stride_cos_n,
    stride_sin_m,
    stride_sin_n,
    stride_out_batch,
    stride_out_head,
    stride_out_m,
    stride_out_n,
    batch_size,
    head_num,
    seq_len,
    H,
    D,
    HID,
    stride_h,
    stride_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if pid_batch >= batch_size or pid_head >= head_num:
        return

    seq_start = 0
    cur_seq_len = seq_len
    if CU_SEQLENS is not None:
        seq_start = tl.load(CU_SEQLENS + pid_batch)
        cur_seq_len = tl.load(SEQLENS + pid_batch)
    elif seq_len > 0:
        cur_seq_len = seq_len
    else:
        cur_seq_len = seq_len

    if pid_m * BLOCK_M >= cur_seq_len:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < cur_seq_len

    cos_ptrs = COS + (seq_start + offs_m[:, None]) * stride_cos_m + offs_n[None, :] * stride_cos_n
    sin_ptrs = SIN + (seq_start + offs_m[:, None]) * stride_sin_m + offs_n[None, :] * stride_sin_n

    cos = tl.load(cos_ptrs, mask=mask_m[:, None], other=0.0)
    sin = tl.load(sin_ptrs, mask=mask_m[:, None], other=0.0)

    x_base_ptr = X + pid_batch * stride_x_batch + pid_head * stride_x_head
    out_base_ptr = OUT + pid_batch * stride_out_batch + pid_head * stride_out_head

    if INTERLEAVED:
        offs_d = 2 * offs_n
        x_ptr0 = x_base_ptr + offs_m[:, None] * stride_x_m + offs_d[None, :] * stride_x_n
        x_ptr1 = x_base_ptr + offs_m[:, None] * stride_x_m + (offs_d + 1)[None, :] * stride_x_n

        x0 = tl.load(x_ptr0, mask=mask_m[:, None], other=0.0).to(DTYPE)
        x1 = tl.load(x_ptr1, mask=mask_m[:, None], other=0.0).to(DTYPE)

        c = cos
        s = sin if not CONJUGATE else -sin
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c

        tl.store(out_base_ptr + offs_m[:, None] * stride_out_m + offs_d[None, :] * stride_out_n, y0, mask=mask_m[:, None])
        tl.store(out_base_ptr + offs_m[:, None] * stride_out_m + (offs_d + 1)[None, :] * stride_out_n, y1, mask=mask_m[:, None])
    else:
        offs_d0 = offs_n
        offs_d1 = offs_n + HID

        x_ptr0 = x_base_ptr + offs_m[:, None] * stride_x_m + offs_d0[None, :] * stride_x_n
        x_ptr1 = x_base_ptr + offs_m[:, None] * stride_x_m + offs_d1[None, :] * stride_x_n

        x0 = tl.load(x_ptr0, mask=mask_m[:, None], other=0.0).to(DTYPE)
        x1 = tl.load(x_ptr1, mask=mask_m[:, None], other=0.0).to(DTYPE)

        c = cos
        s = sin if not CONJUGATE else -sin
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c

        tl.store(out_base_ptr + offs_m[:, None] * stride_out_m + offs_d0[None, :] * stride_out_n, y0, mask=mask_m[:, None])
        tl.store(out_base_ptr + offs_m[:, None] * stride_out_m + offs_d1[None, :] * stride_out_n, y1, mask=mask_m[:, None])


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = 0,
) -> torch.Tensor:
    assert x.dim() == 4
    batch_size, head_num, seq_len, dim = x.shape
    assert dim % 2 == 0
    assert cos.dim() == 4 and sin.dim() == 4
    interleaved = False
    conjugate = False

    dtype = x.dtype
    if dtype == torch.float16:
        triton_dtype = tl.float16
    elif dtype == torch.float32:
        triton_dtype = tl.float32
    else:
        raise ValueError("Unsupported dtype")

    out = torch.empty_like(x)

    HID = dim // 2
    stride_x_batch = x.stride(0)
    stride_x_head = x.stride(1)
    stride_x_m = x.stride(2)
    stride_x_n = x.stride(3)
    stride_cos_m = cos.stride(2)
    stride_cos_n = cos.stride(3)
    stride_sin_m = sin.stride(2)
    stride_sin_n = sin.stride(3)
    stride_out_batch = out.stride(0)
    stride_out_head = out.stride(1)
    stride_out_m = out.stride(2)
    stride_out_n = out.stride(3)

    BLOCK_M = 32
    BLOCK_N = HID
    grid = (triton.cdiv(batch_size, 1), triton.cdiv(head_num, 1), triton.cdiv(seq_len, BLOCK_M))

    rotary_kernel[grid](
        out,
        x,
        cos,
        sin,
        cu_seqlens,
        None,
        stride_x_batch,
        stride_x_head,
        stride_x_m,
        stride_x_n,
        stride_cos_m,
        stride_cos_n,
        stride_sin_m,
        stride_sin_n,
        stride_out_batch,
        stride_out_head,
        stride_out_m,
        stride_out_n,
        batch_size,
        head_num,
        seq_len,
        None,
        dim,
        HID,
        None,
        None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
        DTYPE=triton_dtype,
    )
    return out


##################################################################################################################################################





import torch



def test_apply_rotary():

    results = {}

    

    # Test case 1: Basic test with fixed sequence length and no interleaving

    batch, seqlen, nheads, headdim = 2, 128, 4, 64

    rotary_dim = 32

    x = torch.randn(batch, seqlen, nheads, headdim, device='cuda')

    cos = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    sin = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    output = apply_rotary(x, cos, sin)

    results['test_case_1'] = output.shape



    # Test case 2: Variable length sequences with interleaving

    total_seqlen, nheads, headdim = 256, 4, 64

    batch = 3

    cu_seqlens = torch.tensor([0, 100, 200, 256], device='cuda')

    max_seqlen = 128

    rotary_dim = 32

    x = torch.randn(total_seqlen, nheads, headdim, device='cuda')

    cos = torch.randn(max_seqlen, rotary_dim // 2, device='cuda')

    sin = torch.randn(max_seqlen, rotary_dim // 2, device='cuda')

    output = apply_rotary(x, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=True)

    results['test_case_2'] = output.shape



    # Test case 3: Conjugate flag enabled

    batch, seqlen, nheads, headdim = 2, 128, 4, 64

    rotary_dim = 32

    x = torch.randn(batch, seqlen, nheads, headdim, device='cuda')

    cos = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    sin = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    output = apply_rotary(x, cos, sin, conjugate=True)

    results['test_case_3'] = output.shape



    # Test case 4: Inplace operation

    batch, seqlen, nheads, headdim = 2, 128, 4, 64

    rotary_dim = 32

    x = torch.randn(batch, seqlen, nheads, headdim, device='cuda')

    cos = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    sin = torch.randn(seqlen, rotary_dim // 2, device='cuda')

    output = apply_rotary(x, cos, sin, inplace=True)

    results['test_case_4'] = output.shape



    return results



result_gold = test_apply_rotary()
