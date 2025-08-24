
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(token_ids_ptr, out_ptr, weight_ptr,
                     stride_tokens_b, stride_tokens_s,
                     stride_out_b, stride_out_s, stride_out_d,
                     stride_weight_v, stride_weight_d,
                     seq_len, vocab_size, hidden_size,
                     BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)           # batch
    pid_n = tl.program_id(1)           # sequence block

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    token_ptr = token_ids_ptr + pid_b * stride_tokens_b + offs_n * stride_tokens_s
    mask_n = offs_n < seq_len
    tok_ids = tl.load(token_ptr, mask=mask_n, other=0)   # [BLOCK_N] int32

    for start_d in range(0, hidden_size, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)          # [BLOCK_D]

        mask_d = offs_d < hidden_size
        mask_w = mask_n[:, None] & mask_d[None, :]

        # weight: [v, h]  => gather[token, :]   => [BLOCK_N, BLOCK_D]
        w_offs = tok_ids[:, None] * stride_weight_v + offs_d[None, :] * stride_weight_d
        emb = tl.load(weight_ptr + w_offs, mask=mask_w, other=0.0)

        # out   : [b, seq, h]
        o_offs = pid_b * stride_out_b + offs_n[:, None] * stride_out_s + offs_d[None, :] * stride_out_d
        tl.store(out_ptr + o_offs, emb, mask=mask_w)


def embedding(token_ids: torch.Tensor, weights: torch.Tensor,
              out: torch.Tensor = None) -> torch.Tensor:
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    elif token_ids.dim() != 2:
        raise ValueError("token_ids must be 1-D or 2-D")

    vocab_size, hidden_size = weights.shape
    batch, seq_len = token_ids.shape

    if weights.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError("weights dtype must be fp16/fp32/bf16")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("token_ids dtype must be int32/int64")

    if out is None:
        out = torch.empty((batch, seq_len, hidden_size),
                          dtype=weights.dtype, device=weights.device)

    token_ids = token_ids.contiguous()
    weights   = weights.contiguous()
    out       = out.contiguous()

    BLOCK_N = 64
    BLOCK_D = min(128, triton.next_power_of_2(hidden_size))

    grid = (batch, triton.cdiv(seq_len, BLOCK_N))

    embedding_kernel[grid](
        token_ids, out, weights,
        token_ids.stride(0), token_ids.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        weights.stride(0), weights.stride(1),
        seq_len, vocab_size, hidden_size,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
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
