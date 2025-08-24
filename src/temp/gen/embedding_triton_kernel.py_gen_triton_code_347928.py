
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(weights_ptr, id_ptr, out_ptr,
                     stride_wd, stride_wn,
                     stride_o0, stride_o1,
                     seq_len, dim,
                     BLOCK_N: tl.constexpr,
                     BLOCK_D: tl.constexpr,
                     BLOCK_NN: tl.constexpr):
    pid_d0 = tl.program_id(0)  # block row
    pid_b  = tl.program_id(1)  # batch index

    # D offsets handled within each program
    offs_d = pid_d0 * BLOCK_D + tl.arange(0, BLOCK_D)

    # per-seq id & out base pointers
    seq_id_ptr  = id_ptr  + pid_b * seq_len
    seq_out_ptr = out_ptr + pid_b * seq_len * stride_o0

    for block_n_start in range(0, seq_len, BLOCK_NN):
        offs_n = block_n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        ids = tl.load(seq_id_ptr + offs_n, mask=mask_n)          # [BLOCK_N]
        mask = (ids >= 0) & mask_n
        ids = tl.where(mask, ids, 0)

        # Row-major weights: shape (dim, vocab)  → stride (stride_wd, stride_wn=1)
        # pointer = &weights[ids, offs_d]
        ptrs = weights_ptr + ids[:, None] * stride_wd + offs_d[None, :]  # [BLOCK_N, BLOCK_D]
        vals = tl.load(ptrs, mask=mask[:, None] & (offs_d[None, :] < dim))

        out_ptrs = seq_out_ptr + (offs_n[:, None] * stride_o0 + offs_d[None, :] * stride_o1)
        tl.store(out_ptrs, vals, mask=mask[:, None] & (offs_d[None, :] < dim))


def embedding(weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    weights: (vocab_size, dim)
    indices: (batch, seq)
    returns: (batch, seq, dim)
    """
    orig_dtype = weights.dtype
    # Triton prefers float32 accumulation
    weights_fp = weights.to(torch.float32)

    vocab, dim = weights_fp.shape
    batch, seq = indices.shape

    BLOCK_D_TRITON = triton.next_power_of_2(dim)

    out = torch.empty((batch, seq, dim), dtype=weights_fp.dtype, device=weights.device)
    stride_wd, stride_wn = weights_fp.stride()
    stride_o0, stride_o1 = out.stride(0), out.stride(2)

    # block sizes
    BLOCK_N = 64
    BLOCK_NN = 128
    grid = (triton.cdiv(dim, BLOCK_D_TRITON), batch)

    embedding_kernel[grid](
        weights_fp, indices, out,
        stride_wd, stride_wn,
        stride_o0, stride_o1,
        seq, dim,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D_TRITON,
        BLOCK_NN=BLOCK_NN,
    )

    return out.to(orig_dtype)


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
