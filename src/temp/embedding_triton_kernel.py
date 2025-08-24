import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    tokens_ptr,           # int32*
    out_ptr,              # weight.dtype*
    weight_ptr,           # weight.dtype*
    seq_len,
    vocab_size,
    n_dim,
    stride_tokens,
    stride_out_n,
    stride_out_d,
    stride_weight_vocab,
    stride_weight_dim,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
):
    pid_seq = tl.program_id(0)  # batch dimension
    offs_d = tl.arange(0, n_dim)

    for block_start in range(0, seq_len, BLOCK_NN):
        cur_block_size = tl.minimum(BLOCK_NN, seq_len - block_start)
        block_token_offs = pid_seq * stride_tokens + block_start + tl.arange(0, BLOCK_N)
        mask_n = tl.arange(0, BLOCK_N) < cur_block_size
        block_tokens = tl.load(tokens_ptr + block_token_offs, mask=mask_n, other=0)

        offs_n = block_start + tl.arange(0, BLOCK_N)[:, None]         # [BLOCK_N, 1]
        offs_w = block_tokens[:, None] * stride_weight_vocab + offs_d[None, :] * stride_weight_dim  # [BLOCK_N, n_dim]

        w_vec = tl.load(weight_ptr + offs_w,
                        mask=(offs_n < seq_len)[:, None] & (offs_d[None, :] < n_dim))

        offs_out = pid_seq * stride_out_n + offs_n * stride_out_d + offs_d[None, :]
        tl.store(out_ptr + offs_out,
                 w_vec,
                 mask=(offs_n < seq_len)[:, None] & (offs_d[None, :] < n_dim))


def embedding(tokens: torch.Tensor,
              weight: torch.Tensor) -> torch.Tensor:
    assert tokens.dim() == 2, "Expected tokens shape (batch, seq)"
    bsz, seq_len = tokens.shape
    vocab_size, n_dim = weight.shape
    assert tokens.dtype in [torch.int32, torch.int64], "tokens must be int32 or int64"
    output = torch.empty((bsz, seq_len, n_dim), dtype=weight.dtype, device=weight.device)

    BLOCK_N = 64
    BLOCK_NN = BLOCK_N
    grid = (bsz,)
    embedding_kernel[grid](
        tokens,
        output,
        weight,
        seq_len,
        vocab_size,
        n_dim,
        tokens.stride(0),
        output.stride(0),
        output.stride(1),
        weight.stride(0),
        weight.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN,
    )
    return output
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
