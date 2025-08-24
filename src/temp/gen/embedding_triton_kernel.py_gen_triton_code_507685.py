
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
@triton.jit
def embedding_kernel(
    ids,           # pointer to token-ids tensor               shape [B, S]   (int64)
    weight,        # pointer to embedding weight matrix        shape [V, D]   (float16 or float32)
    out,           # pointer to output  3-D tensor             shape [B, S, D]
    B,             # int: number of sequences (batch size)
    S,             # int: max sequence length for all sequences
    V,             # int: vocabulary size
    D,             # int: embedding dimension
    stride_ids_0,  # leading stride of   ids: = S
    stride_w_0,    # leading stride of  weight: = D
    stride_out_0,  # leading stride of   out: = S * D
    stride_out_1,  #
    stride_out_2,  #
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(0)  # 1-D grid: each block handles one sequence in the batch

    # each block handles every BLOCK_N tokens, each tid loop handles BLOCK_NN tokens
    for b_base in range(0, S, BLOCK_N):
        ids_offset = pid * stride_ids_0 + b_base
        # Load mask
        n_ids = tl.arange(b_base, b_base + BLOCK_N)
        mask_n = n_ids < S

        # Load token indices
        token_ids = tl.load(ids + ids_offset + tl.arange(0, BLOCK_N), mask=mask_n, other=0)

        # Ensure token_ids in [0, V-1]
        token_ids = tl.maximum(0, token_ids)
        token_ids = tl.minimum(V - 1, token_ids)

        # Iterate over tokens in groups of BLOCK_NN
        for start in range(0, BLOCK_N, BLOCK_NN):
            idx_group = start + tl.arange(0, BLOCK_NN)
            group_mask = mask_n & (idx_group < BLOCK_N)

            # Current token ids for this group
            tid = token_ids[start : start + BLOCK_NN]  # shape [BLOCK_NN]
            outs_idx = pid * stride_out_0 + (b_base + start + tl.arange(0, BLOCK_NN)) * stride_out_1

            # Iterate over the embedding dimension in blocks
            for d_start in range(0, D, BLOCK_DMODEL):
                offs_d = d_start + tl.arange(0, BLOCK_DMODEL)
                mask_d = offs_d < D

                valid_mask = group_mask[:, None] & mask_d[None, :]

                # Weight pointer: address strides: weight[tid, d_offs] = weight + tid * stride_w_0 + offs_d
                weight_ptr = weight + tid[:, None] * stride_w_0 + offs_d[None, :]
                emb_vec = tl.load(weight_ptr, mask=valid_mask, other=0.0)

                # Output pointer: address strides
                output_ptr = out + outs_idx[:, None] * stride_out_2 + offs_d[None, :]
                tl.store(output_ptr, emb_vec, mask=valid_mask)

# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def embedding(
    ids: torch.Tensor,      # [B, S]   long int
    weight: torch.Tensor,   # [V, D]   float16 or float32
) -> torch.Tensor:          # Returns: [B, S, D]
    B, S = ids.shape
    V, D = weight.shape
    device = weight.device
    dtype = weight.dtype

    out = torch.empty((B, S, D), dtype=dtype, device=device)

    BLOCK_N = 64
    BLOCK_NN = 64
    BLOCK_DMODEL = triton.next_power_of_2(D)

    grid = (B,)

    embedding_kernel[grid](
        ids,                     # int64
        weight,                  # fp16 / fp32
        out,                     # fp16 / fp32
        B,
        S,
        V,
        D,
        ids.stride(0),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
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
