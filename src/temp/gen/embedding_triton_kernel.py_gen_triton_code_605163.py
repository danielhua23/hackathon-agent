
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(weight_ptr,
                     ids_ptr,
                     out_ptr,
                     stride_w,
                     stride_out,
                     num_tokens,
                     BLOCK_N: tl.constexpr,
                     BLOCK_NN: tl.constexpr,
                     BLOCK_DMODEL: tl.constexpr):
    pid = tl.program_id(0)        # token block
    pid_d = tl.program_id(1)      # d-model block
    start_token = pid * BLOCK_NN

    ids = tl.load(ids_ptr + start_token + tl.arange(0, BLOCK_NN))
    offs_d = pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

    mask_tokens = start_token + tl.arange(0, BLOCK_NN) < num_tokens
    mask_d = offs_d < stride_w

    for i in range(BLOCK_NN):
        if not mask_tokens[i]:
            break
        token_id = ids[i]
        w_off = token_id * stride_w + offs_d
        vec = tl.load(weight_ptr + w_off, mask=mask_d)
        o_off = (start_token + i) * stride_out + offs_d
        tl.store(out_ptr + o_off, vec, mask=mask_d)


def embedding(weight: torch.Tensor,
              ids: torch.Tensor,
              out: torch.Tensor,
              BLOCK_N: int = 1,
              BLOCK_NN: int = 32,
              BLOCK_DMODEL: int = None):
    assert weight.ndim == 2
    assert ids.ndim == 1
    assert out.ndim == 2
    assert out.shape == (ids.shape[0], weight.shape[1])

    _, d_model = weight.shape
    num_tokens = ids.numel()

    if BLOCK_DMODEL is None:
        BLOCK_DMODEL = triton.next_power_of_2(d_model)

    grid = (triton.cdiv(num_tokens, BLOCK_NN),
            triton.cdiv(d_model, BLOCK_DMODEL))
    embedding_kernel[grid](
        weight,
        ids,
        out,
        stride_w=weight.stride(0),
        stride_out=out.stride(0),
        num_tokens=num_tokens,
        BLOCK_N=BLOCK_N,
        BLOCK_NN=BLOCK_NN,
        BLOCK_DMODEL=BLOCK_DMODEL
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
