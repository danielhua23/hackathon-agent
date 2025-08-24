
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
    SEQLEN_OFFSETS,
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    if not IS_VARLEN:
        current_batch_offset = pid_batch * stride_x_batch + pid_head * stride_x_nheads
        X_ptr = X + current_batch_offset
        OUT_ptr = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
        seq_len = seqlen
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seq_len = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X_ptr = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT_ptr = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seq_len:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rk_half = rk % (rotary_dim // 2)

    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rotary_half = rotary_dim // 2

    if not INTERLEAVED:
        k0 = rk_half
        k1 = k0 + rotary_half
        mask_m = rm < seq_len
        mask_m_cs = rm_cs < seqlen_ro

        # first half
        offset0 = rm[:, None] * stride_x_seqlen + k0[None, :] * stride_x_headdim
        x0 = tl.load(X_ptr + offset0, mask=mask_m[:, None] & (k0[None, :] < rotary_half)).to(tl.float32)
        cos0 = tl.load(COS + rm_cs[:, None] * rotary_half + k0[None, :],
                       mask=mask_m_cs[:, None] & (k0[None, :] < rotary_half), other=1.0).to(tl.float32)
        sin0 = tl.load(SIN + rm_cs[:, None] * rotary_half + k0[None, :],
                       mask=mask_m_cs[:, None] & (k0[None, :] < rotary_half), other=0.0).to(tl.float32)

        # second half
        offset1 = rm[:, None] * stride_x_seqlen + k1[None, :] * stride_x_headdim
        x1 = tl.load(X_ptr + offset1, mask=mask_m[:, None] & (k1[None, :] < rotary_dim)).to(tl.float32)

        if CONJUGATE:
            sin0 = -sin0
        o0 = x0 * cos0 - x1 * sin0
        o1 = x0 * sin0 + x1 * cos0

        tl.store(OUT_ptr + offset0, o0, mask=mask_m[:, None] & (k0[None, :] < rotary_half))
        tl.store(OUT_ptr + offset1, o1, mask=mask_m[:, None] & (k1[None, :] < rotary_dim))
    else:
        rk_half = rk // 2
        mask_m = rm < seq_len
        mask_m_cs = rm_cs < seqlen_ro

        x_offsets = rm[:, None] * stride_x_seqlen + rk[None, :] * stride_out_headdim
        cos_sin_offsets = rm_cs[:, None] * rotary_half + rk_half[None, :]

        x = tl.load(X_ptr + x_offsets, mask=mask_m[:, None] & (rk[None, :] < rotary_dim)).to(tl.float32)

        cos = tl.load(COS + cos_sin_offsets,
                      mask=mask_m_cs[:, None] & (rk_half[None, :] < rotary_half), other=1.0).to(tl.float32)
        sin = tl.load(SIN + cos_sin_offsets,
                      mask=mask_m_cs[:, None] & (rk_half[None, :] < rotary_half), other=0.0).to(tl.float32)

        if CONJUGATE:
            sin = -sin

        x0 = tl.where((rk[None, :] % 2) == 0, x, 0)
        x1 = tl.where((rk[None, :] % 2) == 1, x, 0)

        out = x0 * cos + x1 * sin
        tl.store(OUT_ptr + x_offsets, out, mask=mask_m[:, None] & (rk[None, :] < rotary_dim))


from typing import Union, Optional


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None

    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen

    seqlen_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim
    assert headdim <= 256

    if not isinstance(seqlen_offsets, torch.Tensor):
        assert isinstance(seqlen_offsets, int) and seqlen_offsets + seqlen <= seqlen_ro
    else:
        assert seqlen_offsets.shape == (batch,)
        seqlen_offsets = seqlen_offsets.to(torch.int32)

    cos = cos.contiguous()
    sin = sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        seqlen_offsets = seqlen_offsets.contiguous()

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = max(32, triton.next_power_of_2(rotary_dim))
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)

    with torch.cuda.device(x.device.type):
        rotary_kernel[grid](
            output,
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,
            nheads,
            rotary_dim,
            seqlen_ro,
            seqlen // 128,
            output.stride(0) if not is_varlen else 0,
            output.stride(-3),
            output.stride(-2),
            output.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3),
            x.stride(-2),
            x.stride(-1),
            BLOCK_K=BLOCK_K,
            IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
            IS_VARLEN=is_varlen,
            INTERLEAVED=interleaved,
            CONJUGATE=conjugate,
            BLOCK_M=BLOCK_M,
        )
    return output


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
