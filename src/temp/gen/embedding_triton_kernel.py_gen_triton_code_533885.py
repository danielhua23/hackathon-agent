
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(indexes, weight, out, stride_idx_b,
                     stride_idx_s, stride_wt_v, stride_wd,
                     stride_ot_b, stride_ot_s, stride_ot_d,
                     VOCAB_SIZE: tl.constexpr, D_MODEL: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    seq_start = pid_s * BLOCK_N
    d_start = pid_d * BLOCK_NN

    idx_base = pid_b * stride_idx_b + seq_start * stride_idx_s
    valid_seq_len = tl.load(indexes + idx_base)
    valid_seq_len = tl.minimum(valid_seq_len, BLOCK_N)

    offs_s = seq_start + tl.arange(0, BLOCK_N)
    offs_d = d_start + tl.arange(0, BLOCK_NN)

    mask_seq = offs_s < (seq_start + valid_seq_len)
    mask_d = offs_d < D_MODEL

    idx_ptr = indexes + idx_base + offs_s * stride_idx_s
    token_ids = tl.load(idx_ptr, mask=mask_seq, other=0)

    w_offs = (token_ids[:, None] * stride_wt_v) + (offs_d[None, :] * stride_wd)
    emb_vec = tl.load(weight + w_offs, mask=mask_seq[:, None] & mask_d[None, :], other=0.0)

    o_offs = (pid_b * stride_ot_b) + (offs_s * stride_ot_s)[:, None] + (offs_d * stride_ot_d)[None, :]
    tl.store(out + o_offs, emb_vec, mask=mask_seq[:, None] & mask_d[None, :])


def embedding(indexes: torch.Tensor, weight: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    B, S = indexes.shape
    VOCAB_SIZE, D_MODEL = weight.shape

    out = torch.empty((B, S, D_MODEL), dtype=weight.dtype, device=weight.device) if out is None else out

    BLOCK_N = 64
    BLOCK_NN = min(64, triton.next_power_of_2(D_MODEL))

    assert indexes.is_contiguous()
    assert weight.is_contiguous()
    assert out.is_contiguous()

    grid = (B, triton.cdiv(S, BLOCK_N), triton.cdiv(D_MODEL, BLOCK_NN))

    embedding_kernel[grid](
        indexes, weight, out,
        indexes.stride(0), indexes.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        VOCAB_SIZE=VOCAB_SIZE,
        D_MODEL=D_MODEL,
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN
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
