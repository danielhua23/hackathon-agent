
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    token_ids,           # [B, L]
    weight,              # [V, D]
    out,                 # [B, L, D]
    stride_b, stride_l, stride_v, stride_d,
    BLOCK_N: tl.constexpr, BLOCK_NN: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)

    # Offset for D dimension in weight and out
    cols_d = pid_dim * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)
    mask_d = cols_d < stride_d

    # Iterate over sequence positions in strides of BLOCK_NN
    for start_l in tl.range(0, stride_l, BLOCK_NN):
        # Global sequence index
        cols_l = start_l + tl.arange(0, BLOCK_NN)[:, None]  # [BLOCK_NN, 1]

        # Build batch * sequence pointers to token_ids
        off_ids = pid_batch * stride_b + cols_l  # [BLOCK_NN, 1]
        mask_l = cols_l < stride_l
        ids = tl.load(token_ids + off_ids, mask=mask_l)    # [BLOCK_NN, 1]

        # Each id identifies a row in weight
        # Compute pointer into weight: rows=[ids], cols=[cols_d]
        off_weight = ids * stride_v + cols_d      # [BLOCK_NN, BLOCK_DMODEL]
        vals = tl.load(weight + off_weight, mask=mask_l & mask_d)

        # Store into out: [B, L, D]
        off_out = (
            pid_batch * stride_b          # batch stride
            + cols_l      * stride_l      # seq stride
            + cols_d      * 1            # dim stride
        )
        tl.store(out + off_out, vals, mask=mask_l & mask_d)


def embedding(token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert token_ids.is_cuda or token_ids.device.type == "cuda", "inputs should live on GPU"
    assert weight.dim() == 2
    V, D = weight.shape
    B, L = token_ids.shape
    out = torch.empty((B, L, D), dtype=weight.dtype, device=weight.device)

    BLOCK_DMODEL = triton.next_power_of_2(D)
    BLOCK_N = 32
    BLOCK_NN = triton.next_power_of_2(256)  # load up to 256 sequence positions per program

    grid = lambda META: (B, triton.cdiv(D, META["BLOCK_DMODEL"]))

    embedding_kernel[grid](
        token_ids,
        weight,
        out,
        token_ids.stride(0), token_ids.stride(1), weight.stride(0), weight.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_NN=BLOCK_NN, BLOCK_DMODEL=BLOCK_DMODEL
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
