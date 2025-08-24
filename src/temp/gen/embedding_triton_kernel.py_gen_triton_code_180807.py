
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    ids,            # pointer to the seq-len length token-id vector, [batch, seq-len]
    weight,         # pointer to the embedding table, [vocab, d]
    out,            # pointer to the output embeddings, [batch, seq-len, d]
    stride_ids_b,   # stride(ids, 0)
    stride_ids_s,   # stride(ids, 1)
    stride_weight_v, # stride(weight, 0)
    stride_weight_d, # stride(weight, 1)
    stride_out_b,   # stride(out, 0)
    stride_out_s,   # stride(out, 1)
    stride_out_d,   # stride(out, 2)
    vocab_size,
    d,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1) * BLOCK_N
    pid_d = tl.program_id(2) * BLOCK_DMODEL

    # row-major iteration
    for n_base in range(0, BLOCK_N, BLOCK_NN):
        # offset into the tokens
        offsets_s = pid_s + n_base + tl.arange(0, BLOCK_NN)      # [BLOCK_NN]
        mask_s = offsets_s < d        # valid mask over seq-len

        # load token ids  [BLOCK_NN]
        ids_ptr = ids + pid_b * stride_ids_b + offsets_s * stride_ids_s
        cur_ids = tl.load(ids_ptr, mask=mask_s, other=0)

        # mask valid indices in vocab range
        mask_vocab = (cur_ids >= 0) & (cur_ids < vocab_size)

        # Embed over feature dimension
        for d_base in range(0, BLOCK_DMODEL, BLOCK_DMODEL):
            offsets_d = pid_d + d_base + tl.arange(0, BLOCK_DMODEL)  # [BLOCK_DMODEL]
            mask_d = offsets_d < d

            # compute weight ptrs
            weight_ptrs = (
                weight
                + cur_ids[:, None] * stride_weight_v          # [BLOCK_NN, 1] * stride
                + offsets_d[None, :] * stride_weight_d        # [1, BLOCK_DMODEL]
            )
            weight_vals = tl.load(
                weight_ptrs,
                mask=mask_s[:, None] & mask_d[None, :] & mask_vocab[:, None],
                other=0.0
            )
            out_ptrs = (
                out
                + pid_b * stride_out_b
                + offsets_s[:, None] * stride_out_s
                + offsets_d[None, :] * stride_out_d
            )
            tl.store(
                out_ptrs,
                weight_vals,
                mask=mask_s[:, None] & mask_d[None, :]
            )


def embedding(
    ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Wrapper function that launches the Triton embedding_kernel.
    Args:
        ids:   Tensor of token indices, shape [batch, seq-len]
        weight: Embedding weight matrix, shape [vocab_size, d]
    Returns:
        out: Tensor of embeddings, shape [batch, seq-len, d]
    """
    assert ids.dtype in (torch.int64, torch.int32), "ids must be int64/int32"
    assert len(ids.shape) == 2          # [batch, seq-len]
    assert len(weight.shape) == 2       # [vocab, d]

    batch, seq_len = ids.shape
    vocab_size, d = weight.shape

    out = torch.empty((batch, seq_len, d), dtype=weight.dtype, device=weight.device)

    # Simple heuristics
    BLOCK_N  = 64
    BLOCK_NN = 16
    BLOCK_DMODEL = 64

    grid = (
        batch,
        triton.cdiv(seq_len, BLOCK_N),
        triton.cdiv(d, BLOCK_DMODEL)
    )

    embedding_kernel[grid](
        ids,
        weight,
        out,
        ids.stride(0),
        ids.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        vocab_size,
        d,
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN,
        BLOCK_DMODEL=BLOCK_DMODEL,
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
