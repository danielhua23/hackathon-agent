
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    token_ids_ptr,        # pointer to 1-D int32 token_ids
    weight_ptr,           # pointer to 2-D float weights  (vocab, d_model)
    out_ptr,              # pointer to 2-D output         (seq_len, d_model)
    seq_len,              # int
    d_model,              # int
    vob_start_id,         # int
    vob_end_id,           # int
    stride_id,            # int
    stride_w0,            # int
    stride_w1,            # int
    stride_out0,          # int
    stride_out1,          # int
    BLOCK_DMODEL: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_col = tl.program_id(1)

    if pid_seq >= seq_len:
        return

    # token id for this sequence position
    token_id = tl.load(token_ids_ptr + pid_seq * stride_id)
    valid = (token_id >= vob_start_id) & (token_id < vob_end_id)
    tok_id_off = (token_id - vob_start_id) * stride_w0   # row offset

    cols = pid_col * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)
    d_mask = cols < d_model

    weight_off = tok_id_off + cols * stride_w1
    val = tl.load(weight_ptr + weight_off, mask=d_mask & valid, other=0.0)

    out_off = pid_seq * stride_out0 + cols * stride_out1
    tl.store(out_ptr + out_off, val, mask=d_mask)


def embedding(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    vob_start_id: int,
    vob_end_id: int,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Extract embedding vectors for each token-id in `token_ids` using 2-D matrix `weight`.

    token_ids : torch.int32 shape(SEQ_LEN,) with values in [vob_start_id, vob_end_id)
    weight    : weight[vocab, d_model]  (slice assumed to start at vocab 0)
    """

    seq_len, = token_ids.shape
    vocab_size, d_model = weight.shape
    assert token_ids.dtype == torch.int32
    assert weight.dtype in (torch.float16, torch.float32)
    assert weight.is_contiguous()

    if out is None:
        out = torch.empty((seq_len, d_model), dtype=weight.dtype, device=weight.device)

    BLOCK_DMODEL = triton.next_power_of_2(d_model)

    grid = (seq_len, triton.cdiv(d_model, BLOCK_DMODEL))

    embedding_kernel[grid](
        token_ids,
        weight,
        out,
        seq_len,
        d_model,
        vob_start_id,
        vob_end_id,
        token_ids.stride(0) if token_ids.dim() == 1 else 1,
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
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
