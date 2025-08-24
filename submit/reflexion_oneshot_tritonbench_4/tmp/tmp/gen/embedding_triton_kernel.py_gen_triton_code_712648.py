import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(out, indices, weight, seq_len, stride_outb, stride_outm, stride_outd, stride_indb, stride_indm, stride_wem, stride_wd, BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    begin = pid_m * BLOCK_N
    offs_m = begin + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_len = tl.load(seq_len + pid_b)
    mask_m = offs_m < cur_len
    ind_offs = pid_b * stride_indb + offs_m * stride_indm
    ids = tl.load(indices + ind_offs, mask=mask_m, other=0)
    w_offs = ids[:, None] * stride_wem + offs_d[None, :] * stride_wd
    embed = tl.load(weight + w_offs, mask=mask_m[:, None])
    o_offs = pid_b * stride_outb + offs_m[:, None] * stride_outm + offs_d[None, :] * stride_outd
    tl.store(out + o_offs, embed, mask=mask_m[:, None])

def embedding(indices: torch.Tensor, weight: torch.Tensor, vob_start_id: int, vob_end_id: int, out: torch.Tensor) -> None:
    """
    Extract embeddings given token indices via Triton kernel.

    Parameters
    ==========
    indices        (B, M)   int32|int64 token indices
    weight         (V, D)   float16|float32 embedding table
    vob_start_id   int      (reserved, unused)
    vob_end_id     int      (reserved, unused)
    out            (B, M, D) same dtype as weight output buffer
    """
    if indices.ndim == 1:
        indices = indices.unsqueeze(0)
    if out.ndim == 2:
        out = out.unsqueeze(0)
    assert weight.dtype == out.dtype
    assert indices.dtype in (torch.int32, torch.int64)
    assert out.shape[0] == indices.shape[0]
    assert out.shape[1] == indices.shape[1]
    assert out.shape[2] == weight.shape[1]
    B, M = indices.shape
    seq_len_gpu = torch.full((B,), M, dtype=torch.int32, device=indices.device)
    BLOCK_N = 16
    BLOCK_NN = 1
    BLOCK_DMODEL = triton.next_power_of_2(weight.size(1))
    grid = (B, triton.cdiv(M, BLOCK_N))
    embedding_kernel[grid](out, indices, weight, seq_len_gpu, out.stride(0), out.stride(1), out.stride(2), indices.stride(0), indices.stride(1), weight.stride(0), weight.stride(1), BLOCK_N=BLOCK_N, BLOCK_NN=BLOCK_NN, BLOCK_DMODEL=BLOCK_DMODEL)

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
