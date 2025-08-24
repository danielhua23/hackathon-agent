
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
@triton.jit
def embedding_kernel(
    ids,           # pointer to 1-D flattened token-ids tensor  shape [B*S] (int64)
    weight,        # pointer to embedding weight matrix         shape [V, D] (float16/32)
    out,           # pointer to output tensor                   shape [B, S, D]
    B,             # int: batch size
    S,             # int: sequence length
    V,             # int: vocab size
    D,             # int: embedding dim
    stride_w_0,    # weight stride 0 = D
    stride_out_0,  # stride for out dim 0 = S*D
    stride_out_1,  # stride for out dim 1 = D
    stride_out_2,  # stride for out dim 2 = 1
    vob_start_id,  # unused
    vob_end_id,    # unused
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(0)  # 1-D grid: one program per batch element
    b_offset = pid * S  # base offset for this sequence in flat ids
    # Each program handles its sequence: S tokens
    for seq_off in range(0, S, BLOCK_N):
        idx_n = seq_off + tl.arange(0, BLOCK_N)
        mask_n = idx_n < S
        # indices into 1-D ids tensor
        ids_idx = b_offset + idx_n
        token_ids = tl.load(ids + ids_idx, mask=mask_n, other=0)

        # Clamp token ids into [0, V-1]
        token_ids = tl.maximum(0, token_ids)
        token_ids = tl.minimum(V-1, token_ids)

        # Group BLOCK_N tokens into BLOCK_NN chunks
        for grp_off in range(0, BLOCK_N, BLOCK_NN):
            gn = grp_off + tl.arange(0, BLOCK_NN)
            mask_gn = (gn < BLOCK_N) & mask_n
            tid = token_ids[grp_off: grp_off + BLOCK_NN]

            out_base = pid * stride_out_0 + (seq_off + grp_off) * stride_out_1
            # Iterate over D in blocks
            for d_off in range(0, D, BLOCK_DMODEL):
                offs_d = d_off + tl.arange(0, BLOCK_DMODEL)
                mask_d = offs_d < D
                mask = mask_gn[:, None] & mask_d[None, :]

                # Weight load: weight[tid, offs_d]
                w_ptr = weight + tid[:, None] * stride_w_0 + offs_d[None, :]
                emb_vec = tl.load(w_ptr, mask=mask, other=0.0)

                # Output store: out[pid, seq_off+grp_off:grp_off+BLOCK_NN, offs_d]
                o_ptr = out + out_base + gn[:, None] * stride_out_1 + d_off + offs_d[None, :]
                tl.store(o_ptr, emb_vec, mask=mask)

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
def embedding(
    ids: torch.Tensor,    # [B, S] or [B*S] int64
    weight: torch.Tensor, # [V, D]
    vob_start_id: int = 0,
    vob_end_id: int = None,
    out: torch.Tensor = None,
) -> torch.Tensor:        # Returns [B, S, D]

    device = weight.device
    dtype = weight.dtype
    V, D = weight.shape

    if ids.dim() == 1:
        # Handle flattened case: expect B*S tokens
        flat_ids = ids
    else:  # dim == 2
        B, S = ids.shape
        flat_ids = ids.view(-1)  # [B*S]

    if vob_end_id is None:
        vob_end_id = V

    if ids.dim() == 1:
        # Infer B and S from flattened tensor
        # Assume first argument already passed as full 1-D sequence (B*S)
        # Use vob_start_id / vob_end_id if given, but not used in kernel
        # Need outer computation to provide true B and S
        raise ValueError(
            "With 1-D ids you must compute/reshape outside to [B,S] or directly provide 2-D [B, S]"
        )
    else:
        B, S = ids.shape
        flat_ids = ids.view(-1)

    if out is None:
        out = torch.empty((B, S, D), dtype=dtype, device=device)

    BLOCK_N = 128
    BLOCK_NN = 64
    BLOCK_DMODEL = triton.next_power_of_2(D)

    grid = (B,)

    embedding_kernel[grid](
        flat_ids,
        weight,
        out,
        B,
        S,
        V,
        D,
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        vob_start_id,
        vob_end_id,
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
