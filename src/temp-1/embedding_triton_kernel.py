
import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    token_ids_ptr,
    weights_ptr,
    out_ptr,
    vob_start_id,
    vob_end_id,
    stride_b,            # token_ids dim-0 stride
    stride_t,            # token_ids dim-1 stride
    stride_weights_v,    # weights dim-0 stride
    stride_weights_h,    # weights dim-1 stride
    stride_out_b,        # out dim-0 stride
    stride_out_t,        # out dim-1 stride
    stride_out_d,        # out dim-2 stride
    HID,                 # total hidden size
    HID_DMODEL_TILE: tl.constexpr
):
    pid_b = tl.program_id(0)   # batch
    pid_t = tl.program_id(1)   # sequence time
    pid_h = tl.program_id(2)   # hidden tile

    seq_off = pid_b * stride_b + pid_t * stride_t
    tid = tl.load(token_ids_ptr + seq_off).to(tl.int32)
    tid = tl.where(tid < vob_start_id, vob_start_id,
                   tl.where(tid >= vob_end_id, vob_end_id - 1, tid))

    offs_hd = pid_h * HID_DMODEL_TILE + tl.arange(0, HID_DMODEL_TILE)
    mask_hd = offs_hd < HID

    w_ptrs = weights_ptr + tid * stride_weights_v + offs_hd * stride_weights_h
    vec = tl.load(w_ptrs, mask=mask_hd, other=0.0)

    o_ptrs = out_ptr + pid_b * stride_out_b + pid_t * stride_out_t + offs_hd * stride_out_d
    tl.store(o_ptrs, vec, mask=mask_hd)

@triton.autotune(
    configs=[
        triton.Config({'HID_DMODEL_TILE': 64},  num_warps=2, num_stages=2),
        triton.Config({'HID_DMODEL_TILE': 128}, num_warps=4, num_stages=2),
        triton.Config({'HID_DMODEL_TILE': 256}, num_warps=8, num_stages=4),
    ],
    key=['HID']
)
@triton.jit
def embedding_kernel_autotuned(
    token_ids_ptr,
    weights_ptr,
    out_ptr,
    vob_start_id,
    vob_end_id,
    stride_b,
    stride_t,
    stride_weights_v,
    stride_weights_h,
    stride_out_b,
    stride_out_t,
    stride_out_d,
    HID,
    HID_DMODEL_TILE: tl.constexpr,
):
    embedding_kernel(
        token_ids_ptr,
        weights_ptr,
        out_ptr,
        vob_start_id,
        vob_end_id,
        stride_b,
        stride_t,
        stride_weights_v,
        stride_weights_h,
        stride_out_b,
        stride_out_t,
        stride_out_d,
        HID,
        HID_DMODEL_TILE=HID_DMODEL_TILE,
    )

def embedding(token_ids: torch.Tensor, weights: torch.Tensor, vob_start_id: int, vob_end_id: int, out: torch.Tensor = None):
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    batch, seq = token_ids.shape
    vocab, hidden = weights.shape

    if out is None:
        out = torch.empty((batch, seq, hidden), dtype=weights.dtype, device=weights.device)
    else:
        assert out.shape == (batch, seq, hidden), "output tensor shape mismatch"

    grid = lambda META: (batch, seq, triton.cdiv(hidden, META['HID_DMODEL_TILE']))

    embedding_kernel_autotuned[grid](
        token_ids, weights, out,
        vob_start_id, vob_end_id,
        token_ids.stride(0), token_ids.stride(1),
        weights.stride(0),   weights.stride(1),
        out.stride(0),       out.stride(1),     out.stride(2),
        hidden,
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
