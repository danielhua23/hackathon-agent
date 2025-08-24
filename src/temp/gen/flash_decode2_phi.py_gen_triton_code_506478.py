
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,           # [batch, head, seq_block_num, D]
    Mid_O_LogExpSum, # [batch, head, seq_block_num]
    Out,             # [batch, head, D]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head  = tl.program_id(1)

    seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int32)
    block_n_size = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    offs_d = tl.arange(0, BLOCK_DMODEL)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    sum_exp = 0.0
    max_logic = -float("inf")

    for block_id in range(0, block_n_size):
        v_ptrs = (
            Mid_O
            + cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + block_id * stride_mid_os
            + offs_d * stride_mid_od
        )
        log_ptr = (
            Mid_O_LogExpSum
            + cur_batch * stride_lse_b
            + cur_head * stride_lse_h
            + block_id * stride_lse_s
        )
        v = tl.load(v_ptrs)
        lse = tl.load(log_ptr)

        new_max = tl.maximum(max_logic, lse)
        scale = tl.exp(max_logic - new_max)
        exp_lse = tl.exp(lse - new_max)

        acc = acc * scale + v.to(tl.float32) * exp_lse
        sum_exp = sum_exp * scale + exp_lse
        max_logic = new_max

    out_ptrs = (
        Out
        + cur_batch * stride_ob
        + cur_head * stride_oh
        + offs_d * stride_od
    )
    final_val = acc / sum_exp
    tl.store(out_ptrs, final_val.to(Out.dtype.element_ty))

def flash_decode_stage2(
    B_Seqlen: torch.Tensor,
    Mid_O: torch.Tensor,
    Mid_O_LogExpSum: torch.Tensor,
    Out: torch.Tensor,
    BLOCK_SEQ: int,
):
    batch, head_num, D = Mid_O.shape[0], Mid_O.shape[1], Mid_O.shape[-1]
    assert D in {16, 32, 64, 128, 256, 512}
    grid = (batch, head_num)

    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        Mid_O,
        Mid_O_LogExpSum,
        Out,
        Mid_O.stride(0),
        Mid_O.stride(1),
        Mid_O.stride(2) if Mid_O.ndim == 4 else 0,
        Mid_O.stride(-1),
        Mid_O_LogExpSum.stride(0),
        Mid_O_LogExpSum.stride(1),
        Mid_O_LogExpSum.stride(2) if Mid_O_LogExpSum.ndim == 3 else 0,
        Out.stride(0),
        Out.stride(1),
        Out.stride(2) if Out.ndim == 3 else Out.stride(-1),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=D,
        num_warps=4,
        num_stages=2,
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
