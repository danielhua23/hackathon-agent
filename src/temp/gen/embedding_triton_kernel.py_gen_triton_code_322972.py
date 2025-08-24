
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    out, ids, weight,
    stride_os, stride_om,
    stride_ws, stride_wm,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(0)
    start_id = pid * BLOCK_NN
    cols_d = tl.arange(0, BLOCK_DMODEL)

    for i in range(0, BLOCK_NN, BLOCK_N):
        rows_seq = start_id + i + tl.arange(0, BLOCK_N)
        mask = rows_seq < N
        ids_off = rows_seq
        token_ids = tl.load(ids + ids_off, mask=mask, other=0)

        w_off = (token_ids[:, None] * stride_ws + cols_d[None, :] * stride_wm)
        x = tl.load(weight + w_off, mask=mask[:, None], other=0.0)

        o_off = rows_seq[:, None] * stride_os + cols_d[None, :] * stride_om
        tl.store(out + o_off, x, mask=mask[:, None])


def embedding(ids: torch.Tensor,
              weight: torch.Tensor,
              vob_start_id=None,
              vob_end_id=None,
              out: torch.Tensor = None) -> torch.Tensor:
    N = ids.numel()
    DMODEL = weight.size(-1)
    if out is None:
        out = torch.empty((N, DMODEL), dtype=weight.dtype, device=weight.device)

    BLOCK_N = 16
    BLOCK_NN = 32
    BLOCK_DMODEL = triton.next_power_of_2(DMODEL)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_NN']),)

    embedding_kernel[grid](
        out, ids, weight,
        out.stride(0), out.stride(1),
        weight.stride(0), weight.stride(1),
        N,
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
