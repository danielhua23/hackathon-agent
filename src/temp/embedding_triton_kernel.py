


import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    input_ids_ptr,
    weight_ptr,
    out_ptr,
    vob_start_id,
    vob_end_id,
    stride_weight,
    stride_out,
    NUM_SEQS,
    NUM_TOKENS_PER_SEQ,
    embedding_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr
):
    pid_0 = tl.program_id(0)  # sequence index
    pid_1 = tl.program_id(1)  # token index with block stride

    # Calculate mask bounds
    seq_mask = pid_0 < NUM_SEQS
    token_start = pid_1 * BLOCK_NN
    token_mask = token_start + tl.arange(0, BLOCK_NN)
    token_mask = token_mask < NUM_TOKENS_PER_SEQ
    full_mask = seq_mask & token_mask

    # Compute base addresses
    seq_offset = pid_0 * NUM_TOKENS_PER_SEQ * stride_out
    token_offset = token_start * stride_out
    base_out_ptr = out_ptr + seq_offset + token_offset
    base_ids_ptr = input_ids_ptr + pid_0 * NUM_TOKENS_PER_SEQ + token_offset // stride_out

    # Load token IDs with mask
    ids = tl.load(base_ids_ptr + tl.arange(0, BLOCK_NN), mask=full_mask, other=vob_start_id - 1)
    mask_ids = (ids >= vob_start_id) & (ids < vob_end_id) & full_mask

    # Vectorize embedding loads/stores by processing blocks of embedding_dim with BLOCK_DMODEL granularity
    for d in range(0, embedding_dim, BLOCK_DMODEL):
        # Create vectorized weight index
        weight_vec_ptr = weight_ptr + ids * stride_weight + d
        
        # Load vectorized embedding data
        weight_vec = tl.load(weight_vec_ptr, mask=mask_ids, other=0.0)
        
        # Compute output pointer with vectorized store
        out_ptr_vec = base_out_ptr + d
        tl.store(out_ptr_vec, weight_vec, mask=full_mask)

def embedding(input_ids, weight, vob_start_id, vob_end_id, out=None):
    """Triton-accelerated embedding lookup function."""
    if out is None:
        out = torch.empty(
            input_ids.shape[0], input_ids.shape[1], weight.shape[1],
            device=input_ids.device, dtype=weight.dtype
        )

    NUM_SEQS, NUM_TOKENS_PER_SEQ = input_ids.shape
    embedding_dim = weight.shape[1]
    
    # Constants optimized from analysis
    BLOCK_DMODEL = triton.next_power_of_2(embedding_dim)
    BLOCK_N = 64
    BLOCK_NN = 1
    
    # Launch kernel grid
    grid = lambda META: (NUM_SEQS, triton.cdiv(NUM_TOKENS_PER_SEQ, META['BLOCK_NN']))
    
    embedding_kernel[grid](
        input_ids, weight, out,
        vob_start_id, vob_end_id,
        weight.stride(0),
        out.stride(1),
        NUM_SEQS,
        NUM_TOKENS_PER_SEQ,
        embedding_dim,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN
    )
    return out

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
