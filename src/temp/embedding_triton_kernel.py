
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    ids,            # int32/int64 tensor – flattened
    weight,         # fp16/fp32 tensor   (V, D)
    out,            # same dtype as weight  (num_tokens, D)
    num_tokens,     # = B * N
    vocab_size,
    embed_dim,
    stride_weight_row,
    stride_out_row,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_token = tl.program_id(0)  # 1-D grid over tokens
    pid_d     = tl.program_id(1)  # 1-D grid over D

    token_offs = pid_token * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_token = token_offs < num_tokens

    id_vals = tl.load(ids + token_offs, mask=mask_token, other=0)

    valid_tokens = (id_vals >= 0) & (id_vals < vocab_size)

    d_start = pid_d * BLOCK_DMODEL
    d_offs  = d_start + tl.arange(0, BLOCK_DMODEL)
    mask_d  = d_offs < embed_dim

    # load
    w_ptr = (
        weight
        + id_vals[:, None] * stride_weight_row
        + d_offs[None, :]
    )
    val_mask = valid_tokens[:, None] & mask_d[None, :]
    vals = tl.load(w_ptr, mask=val_mask, other=0.0)

    # store
    out_ptr = (
        out
        + token_offs[:, None] * stride_out_row
        + d_offs[None, :]
    )
    tl.store(out_ptr, vals, mask=val_mask)


def embedding(ids: torch.Tensor,
              weight: torch.Tensor,
              vob_start_id=None,
              vob_end_id=None,
              out: torch.Tensor = None) -> torch.Tensor:
    """
    Triton embedding lookup for AMD ROCm.
    ids   : (B, N) int32/int64    – *any* 2-D shape
    weight: (V, D) fp16/fp32
    out   : prealloc (B*N, D) or None
    returns (B, N, D)
    """
    assert ids.dtype in (torch.int32, torch.int64)
    B, N = ids.shape
    V, D = weight.shape

    if out is None:
        out = torch.empty((B * N, D), dtype=weight.dtype, device=ids.device)
    else:
        assert out.shape == (B * N, D), f"Expected out.shape=({B*N}, {D}), got {out.shape}"

    num_tokens = B * N
    ids_flat   = ids.view(-1)

    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(min(128, D))

    grid = (
        triton.cdiv(num_tokens, BLOCK_N),
        triton.cdiv(D, BLOCK_D),
    )

    embedding_kernel[grid](
        ids_flat,
        weight,
        out,
        num_tokens,
        V,
        D,
        weight.stride(0),
        out.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_D,
    )
    return out.view(B, N, D)

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
