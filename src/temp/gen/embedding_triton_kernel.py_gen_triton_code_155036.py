
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    token_ids,
    weight,
    out,
    stride_b,
    stride_l,
    stride_v,
    stride_d,
    stride_ob,
    stride_ol,
    vob_start_id,
    vob_end_id,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)

    cols_d = pid_dim * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)
    mask_d = cols_d < stride_d

    for start_l in tl.range(0, stride_l, BLOCK_NN):
        cols_l = start_l + tl.arange(0, BLOCK_NN)
        mask_l = cols_l < stride_l

        offset_ids = pid_batch * stride_b + cols_l
        ids = tl.load(token_ids + offset_ids, mask=mask_l).to(tl.int64)

        ids = tl.where((ids >= vob_start_id) & (ids < vob_end_id), ids, vob_start_id)
        valid_mask = (ids >= vob_start_id) & (ids < vob_end_id)

        offset_weight = ids[:, None] * stride_v + cols_d[None, :]
        vals = tl.load(
            weight + offset_weight,
            mask=mask_l[:, None] & mask_d[None, :] & valid_mask[:, None]
        )

        offset_out = (
            pid_batch * stride_ob +
            cols_l[:, None] * stride_ol +
            cols_d[None, :]
        )
        tl.store(
            out + offset_out,
            vals,
            mask=mask_l[:, None] & mask_d[None, :]
        )


def embedding(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    vob_start_id: int,
    vob_end_id: int,
    out: torch.Tensor
) -> None:
    assert token_ids.device == weight.device == out.device
    assert token_ids.dtype == torch.int64 or token_ids.dtype == torch.int32
    assert weight.ndim == 2 and out.ndim == 3
    B, L = token_ids.shape
    V, D = weight.shape
    assert out.shape == (B, L, D)

    BLOCK_DMODEL = triton.next_power_of_2(D)
    BLOCK_N = 32
    BLOCK_NN = 256
    grid = (B, triton.cdiv(D, BLOCK_DMODEL))

    embedding_kernel[grid](
        token_ids,
        weight,
        out,
        token_ids.stride(0),
        token_ids.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        vob_start_id,
        vob_end_id,
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN,
        BLOCK_DMODEL=BLOCK_DMODEL
    )


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
