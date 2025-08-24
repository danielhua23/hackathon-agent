
import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,
    seqlen,            # int32
    nheads,            # int32
    rotary_dim,        # int32
    seqlen_ro,         # int32
    CACHE_KEY_SEQLEN,  # int32
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    BLOCK_K    : tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR : tl.constexpr,
    IS_VARLEN  : tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE  : tl.constexpr,
    BLOCK_M    : tl.constexpr,
):
    pid_m    = tl.program_id(0)
    pid_batch= tl.program_id(1)
    pid_head = tl.program_id(2)

    rot_half = rotary_dim // 2
    offset_batch = pid_batch * stride_x_batch if IS_VARLEN == 0 else 0
    cu_b = 0
    cur_seqlen = seqlen
    if IS_VARLEN != 0:
        cu_b = tl.load(CU_SEQLENS + pid_batch)
        cur_seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - cu_b
    offset_x_batch = cu_b * stride_x_seqlen + pid_head * stride_x_nheads
    offset_o_batch = cu_b * stride_out_seqlen + pid_head * stride_out_nheads

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < cur_seqlen

    seq_off = tl.load(SEQLEN_OFFSETS + pid_batch) if IS_SEQLEN_OFFSETS_TENSOR else SEQLEN_OFFSETS
    base_t = rm + seq_off

    offs_k = tl.arange(0, BLOCK_K)

    for k_base in range(0, rot_half, BLOCK_K):
        k = k_base + offs_k
        mask_k = k < rot_half

        idx_cos_s = base_t[:, None] * rot_half + k[None, :]
        mask_cs = (base_t[:, None] < seqlen_ro) & mask_k[None, :]
        cos = tl.load(COS + idx_cos_s, mask=mask_cs, other=1.0).to(tl.float32)
        sin = tl.load(SIN + idx_cos_s, mask=mask_cs, other=0.0).to(tl.float32)

        if INTERLEAVED == 0:
            idx0 = rm[:, None] * stride_x_seqlen + (k[None, :] * stride_x_headdim)
            idx1 = rm[:, None] * stride_x_seqlen + ((k[None, :] + rot_half) * stride_x_headdim)
            x0 = tl.load(X + offset_x_batch + idx0,
                         mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            x1 = tl.load(X + offset_x_batch + idx1,
                         mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            if CONJUGATE != 0:
                sin = -sin
            y0 = x0 * cos - x1 * sin
            y1 = x0 * sin + x1 * cos
            tl.store(OUT + offset_o_batch + idx0,
                     y0, mask=mask_m[:, None] & mask_k[None, :])
            tl.store(OUT + offset_o_batch + idx1,
                     y1, mask=mask_m[:, None] & mask_k[None, :])
        else:
            idx_even = rm[:, None] * stride_x_seqlen + (2 * k[None, :] * stride_x_headdim)
            idx_odd  = rm[:, None] * stride_x_seqlen + ((2 * k[None, :] + 1) * stride_x_headdim)
            real = tl.load(X + offset_x_batch + idx_even,
                           mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            imag = tl.load(X + offset_x_batch + idx_odd,
                           mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            if CONJUGATE != 0:
                sin = -sin
            new_real = real * cos - imag * sin
            new_imag = real * sin + imag * cos
            tl.store(OUT + offset_o_batch + idx_even,
                     new_real, mask=mask_m[:, None] & mask_k[None, :])
            tl.store(OUT + offset_o_batch + idx_odd,
                     new_imag, mask=mask_m[:, None] & mask_k[None, :])

    for k_base in range(rotary_dim, stride_x_headdim, BLOCK_K):
        k = k_base + offs_k
        mask_k = k < stride_x_headdim
        idx = rm[:, None] * stride_x_seqlen + k[None, :] * stride_x_headdim
        val = tl.load(X + offset_x_batch + idx,
                      mask=mask_m[:, None] & mask_k[None, :])
        tl.store(OUT + offset_o_batch + idx,
                 val, mask=mask_m[:, None] & mask_k[None, :])


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
        batch = cu_seqlens.numel() - 1
        seqlen = max_seqlen

    seqlen_ro, rot_half = cos.shape
    rotary_dim = rot_half * 2
    assert rotary_dim <= headdim
    assert seqlen_ro >= seqlen
    assert rotary_dim % 2 == 0
    assert cos.dtype == sin.dtype == x.dtype
    assert headdim <= 512

    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in (torch.int32, torch.int64)
        seqlen_offsets = seqlen_offsets.to(torch.int32).contiguous()
    else:
        assert seqlen + seqlen_offsets <= seqlen_ro

    cu_seqlens_host = None
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
        cu_seqlens_host = cu_seqlens

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = triton.next_power_of_2(min(rotary_dim // 2, 128))

    grid = (triton.cdiv(seqlen, 4), batch, nheads)

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
        0,
        output.stride(-4) if not is_varlen else 0,
        output.stride(-3),
        output.stride(-2),
        output.stride(-1),
        x.stride(-4) if not is_varlen else 0,
        x.stride(-3),
        x.stride(-2),
        x.stride(-1),
        BLOCK_K,
        isinstance(seqlen_offsets, torch.Tensor),
        is_varlen,
        interleaved,
        conjugate,
        4,
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
