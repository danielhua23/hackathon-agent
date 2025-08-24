
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(token_ids_ptr, out_ptr, weight_ptr,
                     stride_tokens_b, stride_tokens_s,
                     stride_out_b, stride_out_s, stride_out_d,
                     stride_weight_v, stride_weight_d,
                     vocab_size: tl.constexpr, hidden_size: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)       # batch dimension
    pid_n = tl.program_id(1)       # sequence-block dimension

    offs_s = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # load token indices for this batch row
    seq_len = 0
    # Token tensor: shape (batch, seq) => stride (seq, 1)
    # We will access: token_ids_ptr += pid_b * stride_tokens_b
    token_ids_row_ptr = token_ids_ptr + pid_b * stride_tokens_b

    # Since seq length is fixed per call from wrapper, assume seq_len is known
    # We'll pass seq_len explicitly via a scalar; instead handle via BLOCK_N mask
    # For now, pass seq_len as a placeholder scalar (not used in kernel after fixing wrapper)

    # Load the tokens for this block
    mask_s = offs_s < stride_tokens_s  # Effective seq_len from wrapper stride storage
    tok_ids = tl.load(token_ids_row_ptr + offs_s, mask=mask_s, other=0)

    for start_d in range(0, hidden_size, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < hidden_size

        # Compute weight offset: [vocab, hidden]
        weight_offs = tok_ids[:, None] * stride_weight_v + offs_d[None, :] * stride_weight_d
        mask_w = mask_s[:, None] & mask_d[None, :]

        emb = tl.load(weight_ptr + weight_offs, mask=mask_w, other=0.0)

        # Compute out offset: [batch, seq, hidden]
        out_offs = pid_b * stride_out_b + offs_s[:, None] * stride_out_s + offs_d[None, :] * stride_out_d
        mask_out = mask_s[:, None] & mask_d[None, :]
        tl.store(out_ptr + out_offs, emb, mask=mask_out)


def embedding(token_ids: torch.Tensor, weights: torch.Tensor, vob_start_id=None, vob_end_id=None,
              out: torch.Tensor = None) -> torch.Tensor:
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    elif token_ids.dim() != 2:
        raise ValueError("token_ids should be 1-D or 2-D tensor")

    vocab_size, hidden_size = weights.shape
    batch, seq = token_ids.shape
    assert weights.dtype in [torch.float16, torch.float32]
    assert token_ids.dtype in [torch.int32, torch.int64]

    if out is None:
        out = torch.empty((batch, seq, hidden_size), dtype=weights.dtype, device=weights.device)

    # Ensure contiguous
    token_ids = token_ids.contiguous()
    weights = weights.contiguous()
    out = out.contiguous()

    BLOCK_N = 64
    BLOCK_D = min(128, triton.next_power_of_2(hidden_size))

    grid = (batch, triton.cdiv(seq, BLOCK_N))

    embedding_kernel[grid](
        token_ids, out, weights,
        int(token_ids.stride(0)), int(seq),
        int(out.stride(0)), int(out.stride(1)), int(out.stride(2)),
        int(weights.stride(0)), int(weights.stride(1)),
        vocab_size, hidden_size,
        BLOCK_N, BLOCK_D
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
