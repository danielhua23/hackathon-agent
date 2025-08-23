import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(weights_ptr, indices_ptr, out_ptr, v_stride_0, v_stride_1, out_stride_0, out_stride_1, vocab_size: tl.constexpr, hidden_size: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr, NUM_SEQ: tl.constexpr, MAX_LEN: tl.constexpr):
    seq_id = tl.program_id(0)
    if seq_id >= NUM_SEQ:
        return
    len_offsets = tl.program_id(1)
    offset_start = len_offsets * BLOCK_NN
    dim_offsets = tl.arange(0, BLOCK_SIZE)
    indices_base = indices_ptr + seq_id * out_stride_0
    out_base = out_ptr + seq_id * out_stride_0
    for i in range(0, BLOCK_NN, BLOCK_N):
        cur_offset = offset_start + i
        cur_mask = cur_offset < MAX_LEN
        token_id = tl.load(indices_base + cur_offset * out_stride_1, mask=cur_mask, other=0)
        token_mask = token_id < vocab_size
        valid_mask = cur_mask & token_mask
        selector = token_id[:, None] * v_stride_0 + dim_offsets[None, :] * v_stride_1
        w_ptrs = weights_ptr + selector
        vec = tl.load(w_ptrs, mask=valid_mask[:, None], other=0.0)
        out_ptrs = out_base + cur_offset[:, None] * out_stride_1 + dim_offsets[None, :]
        tl.store(out_ptrs, vec, mask=valid_mask[:, None])

def embedding(weights: torch.Tensor, indices: torch.Tensor, vob_start_id=None, vob_end_id=None, out=None) -> torch.Tensor:
    assert weights.dim() == 2, 'weights must be 2-D [vocab_size, hidden_size]'
    vocab_size, hidden_size = weights.shape
    assert indices.dim() == 2, 'indices must be 2-D: [batch, seq]'
    batch, seq_len = indices.shape
    if out is None:
        out = torch.empty((batch, seq_len, hidden_size), dtype=weights.dtype, device=weights.device)
    else:
        assert out.shape == (batch, seq_len, hidden_size), f'out tensor shape mismatch: expected {(batch, seq_len, hidden_size)}, got {out.shape}'
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    BLOCK_N = 1
    BLOCK_NN = triton.next_power_of_2(seq_len) if seq_len > 0 else 1
    grid = (batch, triton.cdiv(seq_len, BLOCK_NN))
    embedding_kernel[grid](weights, indices, out, weights.stride(0), weights.stride(1), out.stride(0), out.stride(1), vocab_size=vocab_size, hidden_size=hidden_size, BLOCK_SIZE=BLOCK_SIZE, BLOCK_N=BLOCK_N, BLOCK_NN=BLOCK_NN, NUM_SEQ=batch, MAX_LEN=seq_len)
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
