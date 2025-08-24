
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    token_ids_ptr,        # int32*
    weight_ptr,           # fp16/fp32*
    out_ptr,              # fp16/fp32*
    seq_len,              # int32
    num_tokens,           # int32
    d_model,              # int32
    stride_id,            # int32
    stride_w0,            # int32
    stride_w1,            # int32
    stride_out0,          # int32
    stride_out1,          # int32
    BLOCK_N:    tl.constexpr,
    BLOCK_NN:   tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_seq = tl.program_id(0)  # sequence index
    pid_col = tl.program_id(1)  # d_model block index

    if pid_seq >= seq_len:
        return

    # token dimension indices in this block
    cols = pid_col * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

    # offset into each token’s embedding slice
    d_mask = cols < d_model
    out_offset = pid_seq * stride_out0 + cols * stride_out1
    weight_offset_col = cols * stride_w1

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, num_tokens, BLOCK_N):
        # iterate over tokens in blocks of BLOCK_N
        block_start = start_n
        block_end   = start_n + BLOCK_N
        n_block = tl.arange(block_start, block_end)
        n_mask = n_block < num_tokens

        # flat token index = seq * max_tokens_per_seq + token_in_seq
        flat_idx = pid_seq * stride_id + n_block
        token_ids = tl.load(token_ids_ptr + flat_idx, mask=n_mask, other=0)

        # gather weight rows: token_ids[BLOCK_N] × lookup[BLOCK_DMODEL]
        for inner in range(0, BLOCK_N, BLOCK_NN):
            inner_start = inner
            inner_end   = inner + BLOCK_NN
            inner_range = inner_start + tl.arange(0, BLOCK_NN)
            mask_inner = (n_block < num_tokens) & (inner_range < BLOCK_N)
            inner_seq_ids = token_ids[inner_range - inner_start] if BLOCK_N > 1 else token_ids

            # load weight rows = inner_seq_ids
            w_offs = inner_seq_ids * stride_w0 + weight_offset_col
            w_vals = tl.load(weight_ptr + w_offs, mask=d_mask & mask_inner, other=0.0)
            acc += w_vals

    # store gathered embedding for this sequence
    acc = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + out_offset, acc, mask=d_mask)


def embedding(
    token_ids: torch.Tensor,  # int32, shape (seq_len, num_tokens)
    weight:    torch.Tensor,  # fp16/fp32, shape (vocab_size, d_model)
    out:       torch.Tensor = None,
) -> torch.Tensor:
    seq_len, num_tokens = token_ids.shape
    _, d_model = weight.shape
    assert token_ids.dtype == torch.int32
    assert weight.dtype in [torch.float16, torch.float32]
    assert weight.is_contiguous()

    if out is None:
        out = torch.empty((seq_len, d_model), dtype=weight.dtype, device=weight.device)

    BLOCK_DMODEL = triton.next_power_of_2(d_model)
    BLOCK_N  = 16
    BLOCK_NN = 8

    grid = (seq_len, triton.cdiv(d_model, BLOCK_DMODEL))

    embedding_kernel[grid](
        token_ids,
        weight,
        out,
        seq_len,
        num_tokens,
        d_model,
        token_ids.stride(0),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
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
