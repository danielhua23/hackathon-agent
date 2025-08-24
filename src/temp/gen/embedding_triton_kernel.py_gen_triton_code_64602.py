
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    out_ptr, ids_ptr, weight_ptr,
    stride_os, stride_om,
    stride_ws, stride_wm,
    N, D_MODEL,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_seq = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_rows = rows_seq < N

    token_ids = tl.load(ids_ptr + rows_seq, mask=mask_rows, other=-1)

    cols_d = tl.arange(0, BLOCK_DMODEL)
    mask_cols = cols_d < D_MODEL

    mask_valid = mask_rows[:, None] & mask_cols[None, :] & (
        token_ids[:, None] >= 0) & (token_ids[:, None] < stride_ws)

    w_ptrs = weight_ptr + (
        token_ids[:, None] * stride_ws + cols_d[None, :] * stride_wm)
    x = tl.load(w_ptrs, mask=mask_valid, other=0.0)

    o_ptrs = out_ptr + (rows_seq[:, None] * stride_os +
                        cols_d[None, :] * stride_om)
    tl.store(o_ptrs, x, mask=mask_valid)


def embedding(ids: torch.Tensor, weight: torch.Tensor,
              vob_start_id: int = 0, vob_end_id: int = None,
              out: torch.Tensor = None) -> torch.Tensor:
    ids = ids.contiguous()
    N = ids.numel()
    D_MODEL = weight.shape[-1]

    if vob_end_id is None:
        vob_end_id = weight.shape[0]

    if out is None:
        out = torch.empty((N, D_MODEL), dtype=weight.dtype,
                          device=weight.device)

    if N == 0:
        return out

    BLOCK_N = 32
    BLOCK_DMODEL = triton.next_power_of_2(D_MODEL)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)

    embedding_kernel[grid](
        out, ids, weight,
        out.stride(0), out.stride(1),
        weight.stride(0), weight.stride(1),
        N, D_MODEL,
        BLOCK_N=BLOCK_N,
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
