
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(weight, out, seq_idx, stride_wm, stride_wd, stride_om, stride_od, stride_s,
                     total_tokens, d_model, seq_len,
                     BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr):
    pid_m = tl.program_id(0)  # sequence index within batch
    pid_n = tl.program_id(1)  # token index within sequence (BLOCK_N stride)
    pid_d = tl.program_id(2)  # feature dimension (BLOCK_DMODEL stride)
    
    # global sequence offset
    offs_seq_m = pid_m * seq_len
    offs_seq_n0 = pid_n * BLOCK_N
    
    # collect BLOCK_N embeddings per step
    for nstart in range(0, BLOCK_N, BLOCK_NN):
        offs_n = nstart + tl.arange(0, BLOCK_NN)      # [BLOCK_NN]
        mask_n = offs_n < BLOCK_N                      # [BLOCK_NN]
        global_n = offs_seq_n0 + offs_n                # [BLOCK_NN]
        mask_seq = global_n < seq_len                  # [BLOCK_NN]
        
        # read token ids (int32)
        offs_ids = seq_idx + offs_seq_m + global_n     # [BLOCK_NN]
        token_ids = tl.load(offs_ids, mask=mask_n & mask_seq, other=-1)   # [BLOCK_NN]
        
        # compute offsets in weight tensor
        # flatten token ids to compute global offsets
        offs_weight = token_ids[:, None] * stride_wm + (pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL))[None, :] * stride_wd
        # load BLOCK_NN * BLOCK_DMODEL elements
        local_weight = tl.load(weight + offs_weight, mask=(token_ids[:, None] >= 0) & (pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL))[None, :] < d_model, other=0.0)  # [BLOCK_NN, BLOCK_DMODEL]
        
        # store to output
        offs_out = (offs_seq_m + global_n)[:, None] * stride_om + (pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL))[None, :] * stride_od
        tl.store(out + offs_out, local_weight, mask=(global_n[:, None] < seq_len) & ((pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL))[None, :] < d_model))

def embedding(weight: torch.Tensor, out: torch.Tensor, seq_idx: torch.Tensor):
    assert weight.ndim == 2, "weight must be 2D: [num_embeddings, embedding_dim]"
    assert seq_idx.ndim == 2, "seq_idx must be 2D: [batch_size, seq_len]"
    assert out.ndim == 3, "out must be 3D: [batch_size, seq_len, embedding_dim]"
    assert weight.dtype == out.dtype, "dtype mismatch between weight and out"
    num_embeddings, d_model = weight.shape
    batch_size, seq_len = seq_idx.shape
    assert out.shape == (batch_size, seq_len, d_model), "out shape mismatch"
    assert seq_idx.dtype == torch.int64 or seq_idx.dtype == torch.int32, "seq_idx must be long/int32"
    
    stride_wm = weight.stride(0)
    stride_wd = weight.stride(1)
    stride_om = out.stride(0)
    stride_od = out.stride(2)
    total_tokens = batch_size * seq_len
    
    BLOCK_DMODEL = triton.next_power_of_2(d_model)
    BLOCK_N = 64
    BLOCK_NN = 8
    
    grid = (
        batch_size,
        triton.cdiv(seq_len, BLOCK_N),
        triton.cdiv(BLOCK_DMODEL, BLOCK_DMODEL),
    )
    
    embedding_kernel[grid](
        weight,
        out,
        seq_idx,
        stride_wm,
        stride_wd,
        stride_om,
        stride_od,
        seq_idx.stride(0),
        total_tokens,
        d_model,
        seq_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN,
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
