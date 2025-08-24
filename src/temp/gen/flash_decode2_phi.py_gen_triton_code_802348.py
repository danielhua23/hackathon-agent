
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,              # [B, H, seq_block_num, D]
    Mid_O_LogExpSum,    # [B, H, seq_block_num]
    Out,                # [B, H, D]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_lse_b,
    stride_mid_lse_h,
    stride_mid_lse_s,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # program ids
    cur_batch = tl.program_id(0)
    cur_head  = tl.program_id(1)

    # load actual sequence length
    cur_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int32)
    block_n_size = (cur_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    # initialize softmax accumulators
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # register mapping across D
    offs_d = tl.arange(0, BLOCK_DMODEL)

    for block_id in range(0, block_n_size):
        tv = tl.load(
            Mid_O
            + cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + block_id * stride_mid_os
            + offs_d * stride_mid_od,
        )

        tlogic = tl.load(
            Mid_O_LogExpSum
            + cur_batch * stride_mid_lse_b
            + cur_head * stride_mid_lse_h
            + block_id * stride_mid_lse_s,
        )

        new_max_logic = tl.maximum(max_logic, tlogic)
        scale = tl.exp(max_logic - new_max_logic)
        exp_logic = tl.exp(tlogic - new_max_logic)

        acc = acc * scale + tv * exp_logic
        sum_exp = sum_exp * scale + exp_logic
        max_logic = new_max_logic

    # avoid possible NaN when every block has identical logic
    acc = acc / sum_exp

    # write final result
    offs_out = (
        cur_batch * stride_ob
        + cur_head * stride_oh
        + offs_d * stride_od
    )
    tl.store(Out + offs_out, acc.to(Out.dtype.element_ty))


def flash_decode_stage2(
    B_Seqlen: torch.Tensor,
    Mid_O: torch.Tensor,
    Mid_O_LogExpSum: torch.Tensor,
    Out: torch.Tensor,
    BLOCK_SEQ: int,
):
    """
    Triton wrapper matching externally generated unit-test signature.

    Parameters:
        B_Seqlen            : [B] int32 — sequence lengths per batch
        Mid_O               : [B, H, seq_block_num, D]
        Mid_O_LogExpSum     : [B, H, seq_block_num]   (accumulated/pre-rotated logit terms)
        Out                 : [B, H, D]               (output to populate)
        BLOCK_SEQ           : int — blocking granularity
    Returns:
        torch.Tensor        : Out view after completion
    """

    B, H, seq_block_num, D = Mid_O.shape
    assert Mid_O_LogExpSum.shape == (B, H, seq_block_num)
    assert B_Seqlen.shape == (B,)
    assert Out.shape == (B, H, D)

    # Kernel compile-time constants taken from tensor shapes
    BLOCK_DMODEL = D

    grid = (B, H)
    num_warps = 4
    num_stages = 2

    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        Mid_O,
        Mid_O_LogExpSum,
        Out,
        Mid_O.stride(0),
        Mid_O.stride(1),
        Mid_O.stride(2),
        Mid_O.stride(3),
        Mid_O_LogExpSum.stride(0),
        Mid_O_LogExpSum.stride(1),
        Mid_O_LogExpSum.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return Out


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
