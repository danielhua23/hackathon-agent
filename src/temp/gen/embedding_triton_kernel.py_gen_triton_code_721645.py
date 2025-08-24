
import torch
import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    ids,              # int32/64 [B, L]
    weight,           # fp*      [V, D]
    out,              # fp*      [B, L, D]
    stride_ids_b,     # tl.constexpr ignores run-time values
    stride_ids_l,
    stride_weight_v,
    stride_weight_d,
    stride_out_b,
    stride_out_l,
    stride_out_d,
    V,
    D,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)         # batch   dim
    pid_l = tl.program_id(1) * BLOCK_L    # seq-len dim
    pid_d = tl.program_id(2) * BLOCK_D    # embed dim

    # Compute ranges
    offs_l = pid_l + tl.arange(0, BLOCK_L)     # [BLOCK_L]
    offs_d = pid_d + tl.arange(0, BLOCK_D)     # [BLOCK_D]

    mask_l = offs_l < ids.shape[1]             # [BLOCK_L]
    mask_d = offs_d < D                        # [BLOCK_D]

    # --- Load token ids for this tile -------------------------------------------------
    ids_ptr = ids + pid_b * stride_ids_b + offs_l * stride_ids_l    # [BLOCK_L]
    idx = tl.load(ids_ptr, mask=mask_l, other=0).to(tl.int32)               # [BLOCK_L]

    # Broadcast ids for weight lookup
    # idx: [BLOCK_L] -> [BLOCK_L, 1]
    idx = idx[:, None]

    # --- Load weight rows -------------------------------------------------------------
    w_ptrs = (
        weight
        + idx * stride_weight_v              # broadcast: [BLOCK_L, 1] * stride
        + offs_d[None, :] * stride_weight_d  # broadcast: [1, BLOCK_D] * stride
    )  # -> [BLOCK_L, BLOCK_D]

    mask_v = (idx >= 0) & (idx < V)          # row-valid mask: [BLOCK_L, 1]
    mask = mask_l[:, None] & mask_d[None, :] & mask_v

    embs = tl.load(w_ptrs, mask=mask, other=0.0)   # [BLOCK_L, BLOCK_D]

    # --- Store into output tensor ------------------------------------------------------
    out_ptrs = (
        out
        + pid_b * stride_out_b
        + offs_l[:, None] * stride_out_l
        + offs_d[None, :] * stride_out_d
    )  # [BLOCK_L, BLOCK_D]

    tl.store(out_ptrs, embs, mask=mask_l[:, None] & mask_d[None, :])


def embedding_forward(
    ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Triton-accelerated embedding lookup.
    ids : [B, L] (int32/int64)
    weight: [V, D]
    returns: [B, L, D]
    """
    assert ids.dtype in {torch.int32, torch.int64}, "ids must be int32/int64"
    assert weight.ndim == 2, "weight should be 2-D: [V, D]"

    B, L = ids.shape
    V, D = weight.shape
    out = torch.empty((B, L, D), dtype=weight.dtype, device=weight.device)

    # choose tile sizes that divide dimensions well
    BLOCK_L = 64
    BLOCK_D = triton.next_power_of_2(D)

    grid = (
        B,
        triton.cdiv(L, BLOCK_L),
        triton.cdiv(D, BLOCK_D),
    )

    embedding_kernel[grid](
        ids,
        weight,
        out,
        ids.stride(0),
        ids.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        V,
        D,
        BLOCK_L=BLOCK_L,
        BLOCK_D=BLOCK_D,
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
