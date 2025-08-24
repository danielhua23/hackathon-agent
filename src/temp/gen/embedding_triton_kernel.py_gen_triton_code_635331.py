
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(token_ids_ptr, out_ptr, weight_ptr,
                     stride_tokens, stride_outb, stride_outs, stride_outd,
                     stride_weightv, stride_weightd,
                     vocab_size: tl.constexpr, hidden_size: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_bn = pid_b * stride_outb
    offs_nn = pid_n * BLOCK_NN

    seq_len = tl.load(token_ids_ptr + pid_b * stride_tokens).to(tl.int32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_tok = start_n + tl.arange(0, BLOCK_N)
        mask_tok = offs_tok < seq_len
        tok_ids = tl.load(token_ids_ptr + offs_bn + offs_tok * stride_tokens, mask=mask_tok, other=0)

        for start_d in range(0, hidden_size, BLOCK_NN):
            offs_d = start_d + tl.arange(0, BLOCK_NN)
            mask_d = offs_d < hidden_size
            offs_weight = tok_ids[:, None] * stride_weightv + offs_d[None, :] * stride_weightd
            mask_weight = mask_tok[:, None] & mask_d[None, :]
            embedded = tl.load(weight_ptr + offs_weight, mask=mask_weight, other=0.0)

            offs_out = (offs_bn + offs_tok[:, None] * stride_outs + offs_d[None, :] * stride_outd)
            mask_out = mask_tok[:, None] & mask_d[None, :]
            tl.store(out_ptr + offs_out, embedded, mask=mask_out)


def embedding(token_ids: torch.Tensor, weights: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    b, seq = token_ids.shape
    vocab_size, hidden_size = weights.shape
    assert weights.dtype in [torch.float16, torch.float32]
    assert token_ids.dtype in [torch.int32, torch.int64]

    if out is None:
        out = torch.empty((b, seq, hidden_size), dtype=weights.dtype, device=weights.device)

    BLOCK_N = 64
    BLOCK_NN = min(128, triton.next_power_of_2(hidden_size))

    grid = (b, triton.cdiv(seq, BLOCK_N))

    embedding_kernel[grid](
        token_ids, out, weights,
        token_ids.stride(0), out.stride(0), out.stride(1), out.stride(2),
        weights.stride(0), weights.stride(1),
        vocab_size, hidden_size,
        BLOCK_N, BLOCK_NN
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
