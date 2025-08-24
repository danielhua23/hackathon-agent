import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(ids, weights, out, B, N, V, D, stride_wd, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < B * N
    token_ids = tl.load(ids + offs_m, mask=mask_m).to(tl.int32)
    valid_tok = token_ids < V
    offs_d = tl.arange(0, BLOCK_K)
    mask_d = offs_d < D
    for mm in range(0, BLOCK_M):
        valid_mm = mask_m & (mm < BLOCK_M)
        tid = tl.load(ids + pid_m * BLOCK_M + mm, mask=valid_mm).to(tl.int32)
        w_offs = tid * stride_wd + offs_d
        emb = tl.load(weights + w_offs, mask=(tid < V) & mask_d, other=0.0)
        out_offs = (pid_m * BLOCK_M + mm) * D + offs_d
        tl.store(out + out_offs, emb, mask=valid_mm & mask_d)

def embedding(ids: torch.Tensor, weights: torch.Tensor, out: torch.Tensor | None=None) -> torch.Tensor:
    assert ids.dtype.is_integral
    B, N = ids.shape
    V, D = weights.shape
    if out is None:
        out = torch.empty((B, N, D), dtype=weights.dtype, device=weights.device)
    BLOCK_M = 128
    grid = (triton.cdiv(B * N, BLOCK_M),)
    BLOCK_K = triton.next_power_of_2(D)
    embedding_kernel[grid](ids.view(-1), weights.view(-1), out.view(-1), B, N, V, D, weights.stride(0), BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, num_warps=4, num_stages=2)
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
