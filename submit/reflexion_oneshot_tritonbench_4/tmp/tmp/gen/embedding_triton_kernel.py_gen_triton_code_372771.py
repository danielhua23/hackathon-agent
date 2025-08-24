import torch
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({'BLOCK_N': 64, 'BLOCK_DMODEL': 32}, num_warps=1, num_stages=1), triton.Config({'BLOCK_N': 128, 'BLOCK_DMODEL': 64}, num_warps=2, num_stages=1), triton.Config({'BLOCK_N': 256, 'BLOCK_DMODEL': 128}, num_warps=4, num_stages=1)], key=['seq_len', 'hidden'])
@triton.jit
def embedding_kernel(out_ptr, weight_ptr, indices_ptr, seq_len, hidden, stride_out_s, stride_out_d, stride_weight_v, stride_weight_d, vob_start_id, vob_end_id, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_n = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)
    mask_n = offs_n < seq_len
    mask_d = offs_d < hidden
    mask_nd = mask_n[:, None] & mask_d[None, :]
    ids = tl.load(indices_ptr + offs_n, mask=mask_n, other=vob_end_id).to(tl.int32)
    mask_valid = (ids >= vob_start_id) & (ids < vob_end_id)
    row = ids - vob_start_id
    w_ptr = weight_ptr + row[:, None] * stride_weight_v + offs_d[None, :] * stride_weight_d
    val = tl.load(w_ptr, mask=mask_valid[:, None] & mask_d[None, :], other=0.0)
    o_ptr = out_ptr + offs_n[:, None] * stride_out_s + offs_d[None, :] * stride_out_d
    tl.store(o_ptr, val, mask=mask_nd)

def embedding(weight: torch.Tensor, indices: torch.Tensor, out: torch.Tensor | None=None, vob_start_id: int=0, vob_end_id: int | None=None) -> torch.Tensor:
    assert weight.device.type in {'cuda', 'hip'}
    assert indices.device == weight.device
    assert weight.ndim == 2
    if indices.ndim == 2:
        indices = indices.flatten()
    vob_end_id = weight.size(0) if vob_end_id is None else vob_end_id
    seq_len = indices.numel()
    hidden = weight.size(1)
    if out is None:
        out = torch.empty((seq_len, hidden), dtype=weight.dtype, device=weight.device)
    else:
        out = out.view(seq_len, hidden)
    assert out.shape == (seq_len, hidden)
    grid = (triton.cdiv(seq_len, 64), triton.cdiv(hidden, 64))
    embedding_kernel[grid](out, weight, indices, seq_len, hidden, out.stride(0), out.stride(1), weight.stride(0), weight.stride(1), vob_start_id, vob_end_id)
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
