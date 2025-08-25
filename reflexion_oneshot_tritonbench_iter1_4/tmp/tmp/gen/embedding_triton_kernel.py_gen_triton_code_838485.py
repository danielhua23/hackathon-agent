import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(weight, input_ids, out, vob_start_id, vob_end_id, stride_weight_row, stride_out_row, n_ctx, hiden_size, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_n = offs_n < n_ctx
    token_ids_raw = tl.load(input_ids + offs_n, mask=mask_n, other=vob_end_id)
    valid_id_mask = (token_ids_raw >= vob_start_id) & (token_ids_raw < vob_end_id)
    token_ids_clamped = tl.where(valid_id_mask, token_ids_raw - vob_start_id, 0)
    offs_vec = token_ids_clamped[:, None] * stride_weight_row + offs_d[None, :]
    load_mask = valid_id_mask[:, None] & (offs_d[None, :] < hiden_size)
    vec = tl.load(weight + offs_vec, mask=load_mask, other=0.0)
    vec = tl.where(valid_id_mask[:, None], vec, 0.0)
    dest_offs = offs_n[:, None] * stride_out_row + offs_d[None, :]
    store_mask = mask_n[:, None] & (offs_d[None, :] < hiden_size)
    tl.store(out + dest_offs, vec, mask=store_mask)

@torch.no_grad()
def embedding(input_ids: torch.Tensor, weight: torch.Tensor, vob_start_id: int, vob_end_id: int, out: torch.Tensor):
    assert input_ids.ndim == 1
    assert weight.ndim == 2
    assert out.ndim == 2 and out.shape[0] == input_ids.shape[0] and (out.shape[1] == weight.shape[1])
    n_ctx = input_ids.shape[0]
    BLOCK_DMODEL = triton.next_power_of_2(weight.shape[1])
    BLOCK_N = 128
    grid = (triton.cdiv(n_ctx, BLOCK_N),)
    embedding_kernel[grid](weight, input_ids, out, vob_start_id, vob_end_id, weight.stride(0), out.stride(0), n_ctx, weight.shape[1], BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL, num_warps=4, num_stages=1)

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
