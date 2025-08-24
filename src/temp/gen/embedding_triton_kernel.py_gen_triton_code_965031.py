
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(input_ids_ptr, weight_ptr, out_ptr,
                     vob_start_id: int, vob_end_id: int,
                     stride_ids_0, stride_ids_1,
                     stride_w_v, stride_w_d,
                     stride_out_0, stride_out_1, stride_out_2,
                     seq_len: int, vocab_size: int, dim: int,
                     BLOCK_N: tl.constexpr,
                     BLOCK_DMODEL: tl.constexpr,
                     BLOCK_NN: tl.constexpr):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_d = pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

    block_offset_id = pid_b * seq_len
    block_offset_out = pid_b * seq_len * stride_out_1

    for j in range(0, seq_len, BLOCK_NN):
        offs_n = j + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        token_ids = tl.load(input_ids_ptr + block_offset_id + offs_n, mask=mask_n)  # BLOCK_N
        valid_mask = (token_ids >= vob_start_id) & (token_ids < vob_end_id) & mask_n
        token_ids_offs = token_ids - vob_start_id

        token_ids_offs = tl.where(valid_mask, token_ids_offs, 0)

        w_ptrs = weight_ptr + token_ids_offs[:, None] * stride_w_v + offs_d[None, :] * stride_w_d
        emb_vec = tl.load(w_ptrs, mask=valid_mask[:, None] & (offs_d[None, :] < dim))

        out_ptrs = out_ptr + block_offset_out + offs_n[:, None] * stride_out_1 + offs_d[None, :] * stride_out_2
        tl.store(out_ptrs, emb_vec, mask=valid_mask[:, None] & (offs_d[None, :] < dim))


def embedding(input_ids: torch.Tensor,
              weight: torch.Tensor,
              vob_start_id: int,
              vob_end_id: int,
              out: torch.Tensor
              ) -> torch.Tensor:
    batch, seq_len = input_ids.shape
    vocab_size, dim = weight.shape
    assert out.shape == (batch, seq_len, dim)
    assert weight.dtype == out.dtype

    stride_ids_0, stride_ids_1 = input_ids.stride()
    stride_w_v, stride_w_d = weight.stride()
    stride_out_0, stride_out_1, stride_out_2 = out.stride()

    BLOCK_DMODEL = triton.next_power_of_2(dim)

    BLOCK_N = 64
    BLOCK_NN = 128
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_DMODEL']), batch)

    embedding_kernel[grid](
        input_ids, weight, out,
        int(vob_start_id), int(vob_end_id),
        stride_ids_0, stride_ids_1,
        stride_w_v, stride_w_d,
        stride_out_0, stride_out_1, stride_out_2,
        int(seq_len), int(vocab_size), int(dim),
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_NN=BLOCK_NN
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
