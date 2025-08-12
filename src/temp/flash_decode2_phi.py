
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    Out,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_lse_b,
    stride_mid_o_lse_h,
    stride_mid_o_lse_s,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    cur_seq_len = tl.load(B_Seqlen + cur_batch)
    block_n_size = (cur_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    offs_d = tl.arange(0, BLOCK_DMODEL)

    for block_n in range(block_n_size):
        tv = tl.load(
            Mid_O
            + cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + block_n * stride_mid_os
            + offs_d * stride_mid_od
        )
        tlogic = tl.load(
            Mid_O_LogExpSum
            + cur_batch * stride_mid_o_lse_b
            + cur_head * stride_mid_o_lse_h
            + block_n * stride_mid_o_lse_s
        )

        new_max_logic = tl.maximum(max_logic, tlogic)
        scale = tl.exp(max_logic - new_max_logic)
        acc = acc * scale
        sum_exp = sum_exp * scale

        att_exp = tl.exp(tlogic - new_max_logic)
        sum_exp += att_exp
        acc += att_exp * tv.to(tl.float32)

        max_logic = new_max_logic

    acc = acc / sum_exp
    tl.store(
        Out
        + cur_batch * stride_ob
        + cur_head * stride_oh
        + offs_d * stride_od,
        acc.to(tl.pointer_type.element_ty(Out)),
    )


def flash_decode_stage2(
    Mid_O: torch.Tensor,
    Mid_O_LogExpSum: torch.Tensor,
    B_Seqlen: torch.Tensor,
    Out: torch.Tensor,
    block_seq: int,
):
    batch, head_num = Out.shape[0], Out.shape[1]
    BLOCK_DMODEL = Out.shape[-1]

    stride_mid_ob = Mid_O.stride(0)
    stride_mid_oh = Mid_O.stride(1)
    stride_mid_os = Mid_O.stride(2)
    stride_mid_od = Mid_O.stride(3)
    stride_mid_o_lse_b = Mid_O_LogExpSum.stride(0)
    stride_mid_o_lse_h = Mid_O_LogExpSum.stride(1)
    stride_mid_o_lse_s = Mid_O_LogExpSum.stride(2)
    stride_ob = Out.stride(0)
    stride_oh = Out.stride(1)
    stride_od = Out.stride(2)

    grid = (batch, head_num)
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        Mid_O,
        Mid_O_LogExpSum,
        Out,
        stride_mid_ob,
        stride_mid_oh,
        stride_mid_os,
        stride_mid_od,
        stride_mid_o_lse_b,
        stride_mid_o_lse_h,
        stride_mid_o_lse_s,
        stride_ob,
        stride_oh,
        stride_od,
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )
    return

##################################################################################################################################################



import torch

# Define the test function
def test_flash_decode_stage2():
    # Define the parameters for different test cases
    batch_size = 2
    head_num = 4
    seq_block_num = 3
    head_dim = 64
    block_seq = 16

    test_cases = {
        "test_case_1": {
            "B_Seqlen": torch.randint(1, seq_block_num * block_seq, (batch_size,), dtype=torch.int32, device='cuda'),
            "mid_out": torch.randn((batch_size, head_num, seq_block_num, head_dim), dtype=torch.float32, device='cuda'),
            "mid_out_logexpsum": torch.randn((batch_size, head_num, seq_block_num), dtype=torch.float32, device='cuda'),
            "Out": torch.zeros((batch_size, head_num, head_dim), dtype=torch.float32, device='cuda'),
            "block_seq": block_seq
        },
        "test_case_2": {
            "B_Seqlen": torch.randint(1, seq_block_num * block_seq, (batch_size,), dtype=torch.int32, device='cuda'),
            "mid_out": torch.randn((batch_size, head_num, seq_block_num, head_dim), dtype=torch.float32, device='cuda'),
            "mid_out_logexpsum": torch.randn((batch_size, head_num, seq_block_num), dtype=torch.float32, device='cuda'),
            "Out": torch.zeros((batch_size, head_num, head_dim), dtype=torch.float32, device='cuda'),
            "block_seq": block_seq + 1  # Different block size
        },
        "test_case_3": {
            "B_Seqlen": torch.randint(1, seq_block_num * block_seq, (batch_size,), dtype=torch.int32, device='cuda'),
            "mid_out": torch.randn((batch_size, head_num, seq_block_num, head_dim), dtype=torch.float32, device='cuda'),
            "mid_out_logexpsum": torch.randn((batch_size, head_num, seq_block_num), dtype=torch.float32, device='cuda'),
            "Out": torch.zeros((batch_size, head_num, head_dim), dtype=torch.float32, device='cuda'),
            "block_seq": block_seq // 2  # Different block size
        },
        "test_case_4": {
            "B_Seqlen": torch.randint(1, seq_block_num * block_seq, (batch_size,), dtype=torch.int32, device='cuda'),
            "mid_out": torch.randn((batch_size, head_num, seq_block_num, head_dim), dtype=torch.float32, device='cuda'),
            "mid_out_logexpsum": torch.randn((batch_size, head_num, seq_block_num), dtype=torch.float32, device='cuda'),
            "Out": torch.zeros((batch_size, head_num, head_dim), dtype=torch.float32, device='cuda'),
            "block_seq": block_seq * 2  # Different block size
        }
    }

    # Execute the function for all test cases
    results = {}
    for key, test_case in test_cases.items():
        flash_decode_stage2(test_case["mid_out"], test_case["mid_out_logexpsum"], test_case["B_Seqlen"], test_case["Out"], test_case["block_seq"])
        results[key] = test_case["Out"]

    return results

# Run the test
result_gold = test_flash_decode_stage2()
