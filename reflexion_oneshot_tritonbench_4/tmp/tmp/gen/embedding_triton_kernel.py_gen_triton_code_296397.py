import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(weight, out, indices, vocab_size, d_model, vob_start_id, stride_out_0, stride_weight_0, BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    for k in range(0, BLOCK_NN, BLOCK_N):
        seq_off = pid * BLOCK_NN + k + tl.arange(0, BLOCK_N)
        mask_seq = seq_off < out.shape[0]
        token_idx_raw = tl.load(indices + seq_off, mask=mask_seq, other=0).to(tl.int32)
        token_idx = token_idx_raw - vob_start_id
        token_idx = tl.where(token_idx >= 0, token_idx, 0)
        token_idx = tl.where(token_idx < vocab_size, token_idx, vocab_size - 1)
        w_offs = token_idx[:, None] * d_model + offs_d[None, :]
        vec = tl.load(weight + w_offs)
        o_offs = seq_off[:, None] * stride_out_0 + offs_d[None, :]
        tl.store(out + o_offs, vec, mask=mask_seq[:, None])

def embedding(weight: torch.Tensor, indices: torch.Tensor, vob_start_id: int, vob_end_id: int, out: torch.Tensor=None) -> torch.Tensor:
    shape_2d = indices.shape
    B, SEQ_LEN = shape_2d
    d_model = weight.size(-1)
    weight = weight.view(-1, d_model)
    if out is None:
        out = torch.empty((B * SEQ_LEN, d_model), dtype=weight.dtype, device=weight.device)
    else:
        out = out.view(B * SEQ_LEN, d_model)
    vocab_sz = vob_end_id - vob_start_id if vob_end_id > vob_start_id else weight.size(0)
    BLOCK_N = 16
    BLOCK_NN = 64
    BLOCK_DMODEL = triton.next_power_of_2(d_model)
    grid = lambda META: (triton.cdiv(out.shape[0], META['BLOCK_NN']),)
    embedding_kernel[grid](weight, out, indices.view(-1), vocab_size=vocab_sz, d_model=d_model, vob_start_id=vob_start_id, stride_out_0=out.stride(0), stride_weight_0=weight.stride(0), BLOCK_N=BLOCK_N, BLOCK_NN=BLOCK_NN, BLOCK_DMODEL=BLOCK_DMODEL)
    return out.view(B, SEQ_LEN, d_model)

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
