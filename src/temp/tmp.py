
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    ids, weight, out,
    stride_ids_n, stride_ids_nn,
    stride_weight_t, stride_weight_d,
    stride_out_n, stride_out_d,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(0)
    ids_ptr = ids + pid * stride_ids_n
    out_ptr = out + pid * stride_out_n

    for start_n in range(0, BLOCK_N, BLOCK_NN):
        offs_n = start_n + tl.arange(0, BLOCK_NN)
        mask_n = offs_n < BLOCK_N
        token_ids = tl.load(ids_ptr + offs_n * stride_ids_nn, mask=mask_n, other=0)

        for start_d in range(0, BLOCK_DMODEL, BLOCK_DMODEL):
            offs_d = start_d + tl.arange(0, BLOCK_DMODEL)
            mask_d = offs_d < BLOCK_DMODEL
            weight_ptrs = weight + token_ids[:, None] * stride_weight_t + offs_d[None, :] * stride_weight_d
            out_ptrs = out_ptr + offs_n[:, None] * stride_out_n + offs_d[None, :] * stride_out_d
            embed = tl.load(weight_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            tl.store(out_ptrs, embed, mask=mask_n[:, None] & mask_d[None, :])


def embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert ids.dim() == 2, "ids must be 2-D tensor"
    assert weight.dim() == 2, "weight must be 2-D tensor"
    batch, seq_len = ids.shape
    vocab_size, d_model = weight.shape
    out = torch.empty((batch, seq_len, d_model), dtype=weight.dtype, device=weight.device)

    BLOCK_N = seq_len
    BLOCK_NN = min(64, seq_len)
    BLOCK_DMODEL = triton.next_power_of_2(d_model)

    grid = (batch,)
    embedding_kernel[grid](
        ids, weight, out,
        ids.stride(0), ids.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN,
        BLOCK_DMODEL=BLOCK_DMODEL,
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
