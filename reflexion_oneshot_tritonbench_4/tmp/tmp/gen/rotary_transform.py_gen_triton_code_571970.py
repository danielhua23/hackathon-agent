import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS, seqlen, nheads, rotary_dim, seqlen_ro, CACHE_KEY_SEQLEN, stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim, stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim, BLOCK_K: tl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr, IS_VARLEN: tl.constexpr, INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2
    if not IS_VARLEN:
        x_batch_stride = stride_x_batch
        ox_batch_stride = stride_out_batch
        offset_b = pid_batch
        actual_seqlen = seqlen
    else:
        seq_start = tl.load(CU_SEQLENS + pid_batch)
        seq_end = tl.load(CU_SEQLENS + pid_batch + 1)
        actual_seqlen = seq_end - seq_start
        x_batch_stride = stride_x_seqlen
        ox_batch_stride = stride_out_seqlen
        offset_b = seq_start
    if pid_m * BLOCK_M >= actual_seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < actual_seqlen
    if IS_SEQLEN_OFFSETS_TENSOR:
        off = tl.load(SEQLEN_OFFSETS + pid_batch)
    else:
        off = SEQLEN_OFFSETS
    rm_cos = rm + off
    rk_half = tl.arange(0, BLOCK_K)
    mask_k_half = rk_half < rotary_dim_half
    X_ptr = X + offset_b * x_batch_stride + pid_head * stride_x_nheads
    OUT_ptr = OUT + offset_b * ox_batch_stride + pid_head * stride_out_nheads
    if not INTERLEAVED:
        x0_ptr = X_ptr + rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim
        x1_ptr = X_ptr + rm[:, None] * stride_x_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_x_headdim
        cos_ptr = COS + rm_cos[:, None] * rotary_dim_half + rk_half[None, :]
        sin_ptr = SIN + rm_cos[:, None] * rotary_dim_half + rk_half[None, :]
        cos = tl.load(cos_ptr, mask=mask_m[:, None] & (rm_cos[:, None] < seqlen_ro) & mask_k_half[None, :], other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr, mask=mask_m[:, None] & (rm_cos[:, None] < seqlen_ro) & mask_k_half[None, :], other=0.0).to(tl.float32)
        x0 = tl.load(x0_ptr, mask=mask_m[:, None] & mask_k_half[None, :], other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr, mask=mask_m[:, None] & mask_k_half[None, :], other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT_ptr + rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim, o0, mask=mask_m[:, None] & mask_k_half[None, :])
        tl.store(OUT_ptr + rm[:, None] * stride_out_seqlen + (rk_half[None, :] + rotary_dim_half) * stride_out_headdim, o1, mask=mask_m[:, None] & mask_k_half[None, :])
    else:
        BLOCK_P = BLOCK_K
        rk = tl.arange(0, BLOCK_P)
        mask_k = rk < rotary_dim
        cos_sin_idx = rk // 2
        cos_sin_mask = cos_sin_idx < rotary_dim_half
        cos_ptr = COS + rm_cos[:, None] * rotary_dim_half + cos_sin_idx[None, :]
        sin_ptr = SIN + rm_cos[:, None] * rotary_dim_half + cos_sin_idx[None, :]
        cos_val = tl.load(cos_ptr, mask=mask_m[:, None] & (rm_cos[:, None] < seqlen_ro) & cos_sin_mask[None, :], other=1.0).to(tl.float32)
        sin_val = tl.load(sin_ptr, mask=mask_m[:, None] & (rm_cos[:, None] < seqlen_ro) & cos_sin_mask[None, :], other=0.0).to(tl.float32)
        x_even_ptr = X_ptr + rm[:, None] * stride_x_seqlen + (2 * (rk // 2))[None, :] * stride_x_headdim
        x_odd_ptr = X_ptr + rm[:, None] * stride_x_seqlen + (2 * (rk // 2) + 1)[None, :] * stride_x_headdim
        x_even = tl.load(x_even_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        x_odd = tl.load(x_odd_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        if CONJUGATE:
            sin_val = -sin_val
        is_even = rk % 2 == 0
        rot_even = x_even * cos_val - x_odd * sin_val
        rot_odd = x_even * sin_val + x_odd * cos_val
        final = tl.where(is_even[None, :], rot_even, rot_odd)
        tl.store(OUT_ptr + rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim, final, mask=mask_m[:, None] & mask_k[None, :])

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[torch.Tensor]=None, max_seqlen: Optional[int]=None, interleaved: bool=False, inplace: bool=False, conjugate: bool=False) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, 'max_seqlen must be provided when cu_seqlens is given.'
        assert x.dim() == 3, 'x must be 3-D for variable-length case (total_seqlen, nheads, headdim)'
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim_half = cos.shape
    assert sin.shape == cos.shape
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen
    assert cos.dtype == sin.dtype, f'Mismatched dtypes cos={cos.dtype}, sin={sin.dtype}'
    assert x.dtype == cos.dtype, f'Mismatched dtypes x={x.dtype}, cos={cos.dtype}'
    cos = cos.contiguous()
    sin = sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in {torch.int32, torch.int64}
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and (not inplace):
        if not is_varlen:
            output[..., rotary_dim:].copy_(x[..., rotary_dim:])
        else:
            output[:, :, rotary_dim:].copy_(x[:, :, rotary_dim:])
    BLOCK_K = max(32, min(256, triton.next_power_of_2(rotary_dim_half)))
    BLOCK_M = 4 if interleaved else 8 if rotary_dim <= 64 else 4
    grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads)
    if not is_varlen:
        stride_x_b = x.stride(0)
        stride_x_s = x.stride(1)
        stride_x_n = x.stride(2)
        stride_x_h = x.stride(3)
        stride_o_b = output.stride(0)
        stride_o_s = output.stride(1)
        stride_o_n = output.stride(2)
        stride_o_h = output.stride(3)
    else:
        stride_x_b = 0
        stride_x_s = x.stride(0)
        stride_x_n = x.stride(1)
        stride_x_h = x.stride(2)
        stride_o_b = 0
        stride_o_s = output.stride(0)
        stride_o_n = output.stride(1)
        stride_o_h = output.stride(2)
    rotary_kernel[grid](output, x, cos, sin, cu_seqlens, seqlen_offsets, seqlen, nheads, rotary_dim, seqlen_ro, 0, stride_o_b, stride_o_s, stride_o_n, stride_o_h, stride_x_b, stride_x_s, stride_x_n, stride_x_h, BLOCK_K=BLOCK_K, IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor), IS_VARLEN=is_varlen, INTERLEAVED=interleaved, CONJUGATE=conjugate, BLOCK_M=BLOCK_M)
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
