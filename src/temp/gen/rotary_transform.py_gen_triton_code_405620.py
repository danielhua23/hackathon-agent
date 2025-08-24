
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    X,
    COS,
    SIN,
    CU_SEQLENS,
    OUT,
    stride_xb,
    stride_xh,
    stride_xn,
    stride_xd,
    stride_cosn,
    stride_cosd,
    stride_sinn,
    stride_sind,
    stride_cu_off,
    stride_ob,
    stride_oh,
    stride_on,
    stride_od,
    nheads,
    seqlen,
    rotary_dim,
    interleaved,
    conjugate,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    IS_EVEN_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    rot_dim_half = rotary_dim // 2

    if CU_SEQLENS is None:
        seq_start = 0
        seq_id = pid_batch
    else:
        seq_start = 0
        if pid_batch > 0:
            seq_start = tl.load(CU_SEQLENS + pid_batch - 1)
        seq_end = tl.load(CU_SEQLENS + pid_batch)
        seq_id = seq_start + pid_m
        if seq_id >= seq_end:
            return

    offset_b = seq_id * stride_xb
    offset_h = pid_head * stride_xh
    offset_n = pid_m * stride_xn
    offset_d = tl.arange(0, BLOCK_K)
    offset_k = tl.arange(0, BLOCK_N)

    # Compute input pointer base for this element
    x_base = X + offset_b + offset_h + offset_n
    # Load input values for rotary dimensions
    if IS_EVEN_K:
        x_rot = tl.load(x_base + offset_d, mask=offset_d < rotary_dim)
    else:
        mask_d = offset_d < rotary_dim
        x_rot = tl.load(x_base + offset_d, mask=mask_d)

    # Compute cosine/sine pointers
    cos_base = COS + seq_id * stride_cosn
    sin_base = SIN + seq_id * stride_sinn

    # Load cosine and sine values
    if IS_EVEN_K:
        cos = tl.load(cos_base + offset_d, mask=offset_d < rotary_dim)
        sin = tl.load(sin_base + offset_d, mask=offset_d < rotary_dim)
    else:
        mask_d = offset_d < rotary_dim
        cos = tl.load(cos_base + offset_d, mask=mask_d)
        sin = tl.load(sin_base + offset_d, mask=mask_d)

    # Split into two halves
    x0 = x_rot[:rot_dim_half] if rotary_dim <= BLOCK_K else x_rot[0:rot_dim_half:2] if interleaved else x_rot[:rot_dim_half]
    x1 = x_rot[rot_dim_half:] if rotary_dim <= BLOCK_K else x_rot[1:rot_dim_half*2:2] if interleaved else x_rot[rot_dim_half:]

    # Gather corresponding cos/sin for each half
    cos0 = cos[:rot_dim_half] if rotary_dim <= BLOCK_K else cos[0:rot_dim_half:2] if interleaved else cos[:rot_dim_half]
    cos1 = cos[rot_dim_half:] if rotary_dim <= BLOCK_K else cos[1:rot_dim_half*2:2] if interleaved else cos[rot_dim_half:]
    sin0 = sin[:rot_dim_half] if rotary_dim <= BLOCK_K else sin[0:rot_dim_half:2] if interleaved else sin[:rot_dim_half]
    sin1 = sin[rot_dim_half:] if rotary_dim <= BLOCK_K else sin[1:rot_dim_half*2:2] if interleaved else sin[rot_dim_half:]

    if conjugate:
        sin0 = -sin0
        sin1 = -sin1

    # Apply rotary transform
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

    # Prepare output pointers
    out_base = OUT + offset_b + offset_h + offset_n

    # Store rotary section
    if interleaved:
        rot_indices = tl.arange(0, rotary_dim)
        # Handle interleaved storage pattern
        if rotary_dim <= BLOCK_K:
            tl.store(out_base + rot_indices[0::2], y0, mask=rot_indices[0::2] < rotary_dim)
            tl.store(out_base + rot_indices[1::2], y1, mask=rot_indices[1::2] < rotary_dim)
        else:
            tl.store(out_base + rot_indices[0::2], y0, mask=rot_indices[0::2] < rotary_dim)
            tl.store(out_base + rot_indices[1::2], y1, mask=rot_indices[1::2] < rotary_dim)
    else:
        if rotary_dim <= BLOCK_K:
            tl.store(out_base + offset_d[:rot_dim_half], y0, mask=offset_d[:rot_dim_half] < rotary_dim)
            tl.store(out_base + offset_d[rot_dim_half:], y1, mask=offset_d[rot_dim_half:] < rotary_dim)
        else:
            tl.store(out_base + offset_d[:rot_dim_half], y0, mask=offset_d[:rot_dim_half] < rotary_dim)
            tl.store(out_base + offset_d[rot_dim_half:], y1, mask=offset_d[rot_dim_half:] < rotary_dim)

    # Copy non-rotary dimensions
    if rotary_dim < BLOCK_K:
        if IS_EVEN_K:
            x_non_rot = tl.load(x_base + offset_d + rotary_dim, mask=offset_d + rotary_dim < BLOCK_K)
            tl.store(out_base + offset_d + rotary_dim, x_non_rot, mask=offset_d + rotary_dim < BLOCK_K)
        else:
            mask_rest = (offset_d + rotary_dim) < BLOCK_K
            x_non_rot = tl.load(x_base + offset_d + rotary_dim, mask=mask_rest)
            tl.store(out_base + offset_d + rotary_dim, x_non_rot, mask=mask_rest)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int = 0,
    cu_seqlens: torch.Tensor = None,
    max_seqlen: int = None,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    assert rotary_dim % 2 == 0
    assert cos.shape == sin.shape

    batch, seqlen = x.shape[0], x.shape[2]
    nheads = x.shape[1]
    
    if max_seqlen is None:
        max_seqlen = seqlen
    
    # Determine block parameters
    BLOCK_K = triton.next_power_of_2(rotary_dim)
    BLOCK_N = triton.next_power_of_2(max_seqlen)
    
    # Grid dimensions
    grid = lambda META: (batch, nheads, triton.cdiv(max_seqlen, META['BLOCK_N']))
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    rotary_kernel[grid](
        x, cos, sin, cu_seqlens, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        cu_seqlens.stride(0) if cu_seqlens is not None else 0,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        nheads, seqlen, rotary_dim, False, False,
        BLOCK_K=BLOCK_K,
        BLOCK_N=min(BLOCK_N, 128),
        IS_EVEN_N=x.shape[2] % min(BLOCK_N, 128) == 0,
        IS_EVEN_K=rotary_dim % BLOCK_K == 0,
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
