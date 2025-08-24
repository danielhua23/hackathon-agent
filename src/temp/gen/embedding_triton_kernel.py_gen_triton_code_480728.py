import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(weight,
                     ids,
                     out,
                     stride_w,
                     stride_out,
                     num_tokens,
                     BLOCK_NN: tl.constexpr,
                     BLOCK_DMODEL: tl.constexpr):
    pid_bn = tl.program_id(0)     # block id over the batch dimension
    pid_d  = tl.program_id(1)     # block id over the d_model dimension

    # offset of token indices handled by this program instance
    offs_n = pid_bn * BLOCK_NN + tl.arange(0, BLOCK_NN)
    mask_n = offs_n < num_tokens                     # mask out-of-bounds tokens
    token_ids = tl.load(ids + offs_n, mask=mask_n)   # block of token-ids

    # offset of feature dimensions handled by this program instance
    offs_d = pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < stride_w                       # stride_w == d_model

    # load d_model vectors, one per active token_id, using broadcasting
    # shape = (BLOCK_NN, BLOCK_DMODEL)
    w_offs = token_ids[:, None] * stride_w + offs_d[None, :]
    vec = tl.load(weight + w_offs, mask=mask_n[:, None] & mask_d[None, :])

    # write to output tensor
    o_offs = offs_n[:, None] * stride_out + offs_d[None, :]
    tl.store(out + o_offs, vec, mask=mask_n[:, None] & mask_d[None, :])


def embedding(ids: torch.Tensor,
              weight: torch.Tensor,
              vob_start_id: int,
              vob_end_id: int,
              out: torch.Tensor,
              BLOCK_NN: int = 32,
              BLOCK_DMODEL: int = None):
    num_tokens = ids.numel()
    _, d_model = weight.shape

    if BLOCK_DMODEL is None:
        BLOCK_DMODEL = triton.next_power_of_2(d_model)

    grid = (
        triton.cdiv(num_tokens, BLOCK_NN),
        triton.cdiv(d_model, BLOCK_DMODEL),
    )

    embedding_kernel[grid](
        weight,
        ids,
        out,
        stride_w=weight.stride(0),
        stride_out=out.stride(0),
        num_tokens=num_tokens,
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
