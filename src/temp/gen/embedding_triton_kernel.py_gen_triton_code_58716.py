
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    ptr_ids,         # 1d raw pointer
    ptr_weight,      # 1d raw pointer
    ptr_out,         # 1d raw pointer
    stride_ids_b,
    stride_ids_l,
    stride_weight_v,
    stride_weight_d,
    stride_out_b,
    stride_out_l,
    stride_out_d,
    vob_start_id,
    B,
    L,
    D,
    V,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l_blk = tl.program_id(1)
    pid_d_blk = tl.program_id(2)

    offs_l = pid_l_blk * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = pid_d_blk * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_l = offs_l < L
    mask_d = offs_d < D

    ids_base = ptr_ids + pid_b * stride_ids_b
    idxs = tl.load(ids_base + offs_l * stride_ids_l,
                   mask=mask_l, other=0).to(tl.int32)
    idxs = idxs - vob_start_id
    idxs = idxs[:, None]

    w_base = ptr_weight
    w_row_stride = stride_weight_v
    w_col_stride = stride_weight_d
    weight_ptrs = w_base + \
        idxs * w_row_stride + offs_d[None, :] * w_col_stride

    mask_v = (idxs >= 0) & (idxs < V)
    mask = mask_l[:, None] & mask_d[None, :] & mask_v

    embs = tl.load(weight_ptrs, mask=mask, other=0.0)

    out_base = ptr_out + pid_b * stride_out_b
    out_ptrs = out_base + \
        offs_l[:, None] * stride_out_l + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, embs, mask=mask)

def embedding(
    ids: torch.Tensor,
    weight: torch.Tensor,
    vob_start_id: int,
    vob_end_id: int,
    out: torch.Tensor,
) -> torch.Tensor:
    assert ids.dtype in (torch.int32, torch.int64)
    assert weight.ndim == 2
    inferred_D = weight.shape[1]
    if out.numel() == 0:
        out = torch.empty((*ids.shape, inferred_D), dtype=weight.dtype, device=weight.device)
    else:
        assert out.shape[:-1] == ids.shape
        assert out.shape[-1] == inferred_D

    B = ids.shape[0]
    L = ids.shape[1] if ids.ndim == 2 else 1
    ids = ids.view(B, L)
    out = out.view(B, L, inferred_D)

    D = inferred_D
    V = vob_end_id - vob_start_id
    assert V <= weight.shape[0]

    BLOCK_L = 64
    BLOCK_D = triton.next_power_of_2(D)

    grid = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(D, BLOCK_D))

    embedding_kernel[grid](
        ids, weight, out,
        ids.stride(0),
        ids.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        vob_start_id,
        B, L, D, V,
        BLOCK_L=BLOCK_L,
        BLOCK_D=BLOCK_D,
    )
    return out


##################################################################################################################################################





import torch



def test_embedding():

    # 参数定义

    vocab_size = 1000         # 词汇表大小

    embedding_dim = 512       # 嵌入维度

    sequence_length = 128     # 输入序列长度

    vob_start_id = 10         # 词汇表起始 ID

    vob_end_id = 1000         # 词汇表结束 ID



    # 创建测试输入张量

    input_ids = torch.randint(

        vob_start_id, vob_end_id, (sequence_length,), dtype=torch.int32, device='cuda'

    )

    weight = torch.randn(

        vocab_size, embedding_dim, dtype=torch.float32, device='cuda'

    )

    out = torch.zeros(

        sequence_length, embedding_dim, dtype=torch.float32, device='cuda'

    )



    # 调用嵌入函数

    embedding(input_ids, weight, vob_start_id, vob_end_id, out)



    # 保存结果

    results = {}

    results['test_case_1'] = out.clone()



    # 测试不同的输入

    input_ids = torch.randint(

        vob_start_id, vob_end_id, (sequence_length,), dtype=torch.int32, device='cuda'

    )

    embedding(input_ids, weight, vob_start_id, vob_end_id, out)

    results['test_case_2'] = out.clone()



    # 测试不同的词汇表范围

    vob_start_id = 0

    vob_end_id = 500

    input_ids = torch.randint(

        vob_start_id, vob_end_id, (sequence_length,), dtype=torch.int32, device='cuda'

    )

    embedding(input_ids, weight, vob_start_id, vob_end_id, out)

    results['test_case_3'] = out.clone()



    # 测试不同的嵌入维度

    embedding_dim = 256

    weight = torch.randn(

        vocab_size, embedding_dim, dtype=torch.float32, device='cuda'

    )

    out = torch.zeros(

        sequence_length, embedding_dim, dtype=torch.float32, device='cuda'

    )

    embedding(input_ids, weight, vob_start_id, vob_end_id, out)

    results['test_case_4'] = out.clone()



    return results



result_gold = test_embedding()
