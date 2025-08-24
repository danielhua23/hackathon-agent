
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLENS_OFFSETS, 
    stride_out_batch, stride_out_head, stride_out_m, stride_out_k,
    stride_x_batch, stride_x_head, stride_x_m, stride_x_k,
    stride_cos_batch, stride_cos_m, stride_cos_k,
    stride_sin_batch, stride_sin_m, stride_sin_k,
    rotary_dim, rotary_half, conjugate,
    HEADS: tl.constexpr, SEQLEN: tl.constexpr, DIM: tl.constexpr,
    IS_VARIABLE: tl.constexpr, INTERLEAVED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    if pid_batch >= stride_out_batch:
        return

    seqlen_offset = 0
    if IS_VARIABLE:
        seqlen_offset = tl.load(SEQLENS_OFFSETS + pid_batch)
        seq_len = tl.load(CU_SEQLENS + pid_batch + 1) - tl.load(CU_SEQLENS + pid_batch)
        if pid_m >= seq_len:
            return
    else:
        if SEQLEN is not None and pid_m >= SEQLEN:
            return
        seqlen_offset = tl.load(SEQLENS_OFFSETS + pid_batch) if SEQLENS_OFFSETS else 0

    rotary_dim = rotary_dim
    k = tl.arange(0, BLOCK_K)

    # Compute offsets for X
    if INTERLEAVED:
        offs_x = (
            pid_batch * stride_x_batch
            + pid_head * stride_x_head
            + pid_m * stride_x_m
            + (k * 2) * stride_x_k
        )
    else:
        offs_x = (
            pid_batch * stride_x_batch
            + pid_head * stride_x_head
            + pid_m * stride_x_m
            + k * stride_x_k
        )

    # Compute offsets for COS/SIN
    offs_cos_sin = pid_m * stride_cos_m + k * stride_cos_k

    # Load COS/SIN
    cos = tl.load(COS + offs_cos_sin, mask=k < rotary_dim, other=1.0)
    sin = tl.load(SIN + offs_cos_sin, mask=k < rotary_dim, other=0.0)

    # Process rotary pairs
    for i in range(0, tl.cdiv(rotary_dim, 2), BLOCK_K // 2):
        # Calculate indices for current pair
        if INTERLEAVED:
            idx = i * 2
            k0 = idx
            k1 = idx + 1
        else:
            idx = i
            k0 = idx
            k1 = idx + rotary_half

        # Load x0, x1
        x0 = tl.load(X + offs_x + k0 * stride_x_k, mask=k0 < rotary_dim, other=0.0)
        x1 = tl.load(X + offs_x + k1 * stride_x_k, mask=k1 < rotary_dim, other=0.0)

        # Apply rotation
        if conjugate:
            out0 = x0 * cos - x1 * sin
            out1 = x0 * sin + x1 * cos
        else:
            out0 = x0 * cos + x1 * sin
            out1 = -x0 * sin + x1 * cos

        # Store results
        tl.store(OUT + offs_x + k0 * stride_x_k, out0, mask=k0 < rotary_dim)
        tl.store(OUT + offs_x + k1 * stride_x_k, out1, mask=k1 < rotary_dim)

    # Handle non-rotary dimensions (copy original values)
    if rotary_dim < DIM:
        for i in range(rotary_dim, DIM, BLOCK_K):
            offs_non_rot = (
                pid_batch * stride_x_batch
                + pid_head * stride_x_head
                + pid_m * stride_x_m
                + i * stride_x_k
            )
            val = tl.load(X + offs_non_rot)
            tl.store(OUT + offs_non_rot, val)


def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlen_offsets: torch.Tensor = None,
    cu_seqlens: torch.Tensor = None, max_seqlen: int = None, interleaved: bool = False,
    in_place: bool = False, conjugate: bool = False
) -> torch.Tensor:
    dims = x.dim()
    assert dims in [3, 4], "Input tensor must be 3D (B, T, D) or 4D (B, H, T, D)"
    
    if dims == 3:  # Treat as (B, T, D)
        batch, seqlen, dim = x.shape
        heads = 1
        x = x.view(batch, heads, seqlen, dim)
    else:  # dims == 4: (B, H, T, D)
        batch, heads, seqlen, dim = x.shape
    
    rotary_dim = cos.shape[-1]
    rotary_half = rotary_dim // 2
    
    assert rotary_dim <= dim, "Rotary dimension must be <= feature dimension"
    assert cos.shape == sin.shape, "COS and SIN must have same shape"
    assert cos.shape[-1] == rotary_dim, "Last dimension of COS/SIN must match rotary_dim"

    # Prepare output tensor
    if in_place:
        out = x
    else:
        out = torch.empty_like(x)

    # Handle max_seqlen for grid dimension
    actual_max_seqlen = max_seqlen if max_seqlen is not None else seqlen
    
    # Prepare sequence offsets
    if seqlen_offsets is None:
        seqlen_offsets = torch.zeros(batch, dtype=torch.int64, device=x.device)
    
    # Determine IS_VARIABLE flag
    IS_VARIABLE = cu_seqlens is not None
    
    # Grid configuration
    grid = lambda META: (
        batch,
        heads,
        triton.cdiv(actual_max_seqlen, META["BLOCK_M"])
    )
    
    # Configure block sizes (tune these parameters)
    BLOCK_M = min(64, actual_max_seqlen)
    BLOCK_K = min(64, rotary_dim)
    
    rotary_kernel[grid](
        out, x, cos, sin, cu_seqlens, seqlen_offsets,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0) if cos.dim() > 1 else 0, cos.stride(-2) if cos.dim() > 1 else 0, cos.stride(-1),
        sin.stride(0) if sin.dim() > 1 else 0, sin.stride(-2) if sin.dim() > 1 else 0, sin.stride(-1),
        rotary_dim, rotary_half, conjugate,
        HEADS=heads, SEQLEN=seqlen, DIM=dim,
        IS_VARIABLE=IS_VARIABLE, INTERLEAVED=interleaved,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
    )
    
    return out.view(batch, seqlen, dim) if dims == 3 else out


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
