
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

    rows_n_full = stride_l
    for start_n in tl.range(0, rows_n_full, BLOCK_NN):
        rows_n = start_n + tl.arange(0, BLOCK_NN)
        mask_n = rows_n < rows_n_full

        offset_ids = pid_batch * stride_b + rows_n
        ids = tl.load(token_ids + offset_ids, mask=mask_n, other=0)

        mask_valid = (ids >= vob_start_id) & (ids < vob_end_id)
        safe_ids = tl.where(mask_valid, ids, 0)

        emb_ptr = weight + (safe_ids[:, None] * stride_v + cols_d[None, :] * 1)
        emb_vals = tl.load(emb_ptr, mask=(mask_n[:, None] & mask_d[None, :]), other=0.0)

        out_ptr = out + (pid_batch * stride_ob + rows_n[:, None] * stride_ol + cols_d[None, :])
        tl.store(out_ptr, emb_vals, mask=(mask_n[:, None] & mask_d[None, :]))


def embedding(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    vob_start_id: int,
    vob_end_id: int,
    out: torch.Tensor
) -> None:
    assert token_ids.dtype in (torch.int64, torch.int32)
    V, D = weight.shape
    B = token_ids.numel() // token_ids.size(-1) if token_ids.ndim > 1 else 1
    L = token_ids.size(-1)

    if token_ids.ndim == 1:
        assert out.numel() == L * D and out.size(-1) == D
    else:
        assert out.numel() == B * L * D and out.size(-1) == D

    BLOCK_DMODEL = triton.next_power_of_2(D)
    BLOCK_N = 32
    BLOCK_NN = 256

    if token_ids.ndim == 1:
        token_ids = token_ids.contiguous()
        out_view = out.view(L, D).contiguous()
        grid = (1, triton.cdiv(D, BLOCK_DMODEL))
        stride_b = 0
        stride_l = L
        stride_ob = 0
        stride_ol = D
    else:
        token_ids = token_ids.view(B, L).contiguous()
        out_view = out.view(B, L, D).contiguous()
        grid = (B, triton.cdiv(D, BLOCK_DMODEL))
        stride_b = token_ids.stride(0)
        stride_l = token_ids.size(-1)
        stride_ob = out_view.stride(0)
        stride_ol = out_view.stride(1)

    stride_v = weight.stride(0)
    stride_d_true = D

    embedding_kernel[grid](
        token_ids,
        weight,
        out_view,
        stride_b,
        stride_l,
        stride_v,
        stride_d_true,
        stride_ob,
        stride_ol,
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
