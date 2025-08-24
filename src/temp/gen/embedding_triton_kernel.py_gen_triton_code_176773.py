
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
    vob_start_id,         # int32
    vob_end_id,           # int32
    stride_id,            # int32
    stride_w0,            # int32
    stride_w1,            # int32
    stride_out0,          # int32
    stride_out1,          # int32
    BLOCK_N:    tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_seq = tl.program_id(0)  # sequence index
    pid_col = tl.program_id(1)  # d_model block index

    if pid_seq >= seq_len:
        return

    cols = pid_col * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)
    d_mask = cols < d_model
    out_offset = pid_seq * stride_out0 + cols * stride_out1
    weight_offset_col = cols * stride_w1

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < num_tokens
    flat_ids = pid_seq * stride_id + n_offsets
    token_ids = tl.load(token_ids_ptr + flat_ids, mask=n_mask, other=0)
    valid = (token_ids >= vob_start_id) & (token_ids < vob_end_id)
    token_ids = token_ids - vob_start_id

    w_offs = token_ids * stride_w0 + weight_offset_col
    w_vals = tl.load(weight_ptr + w_offs, mask=d_mask[:, None] & valid[None, :], other=0.0)
    acc = tl.sum(w_vals, axis=1)

    acc = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + out_offset, acc, mask=d_mask)


def embedding(
    token_ids: torch.Tensor,  # int32, shape (seq_len, num_tokens)
    weight:    torch.Tensor,  # fp16/fp32, shape (vocab_size, d_model)
    vob_start_id: int,
    vob_end_id: int,
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
    BLOCK_N = min(triton.next_power_of_2(num_tokens), 64)

    grid = (seq_len, triton.cdiv(d_model, BLOCK_DMODEL))

    embedding_kernel[grid](
        token_ids,
        weight,
        out,
        seq_len,
        num_tokens,
        d_model,
        vob_start_id,
        vob_end_id,
        token_ids.stride(0),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=BLOCK_N,
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
