
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
    stride_mid_les_b,
    stride_mid_les_h,
    stride_mid_les_s,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    seqlen = tl.load(B_Seqlen + cur_batch)
    block_n_size = tl.maximum(0, (seqlen + BLOCK_SEQ - 1) // BLOCK_SEQ)
    
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    for block_id in range(0, block_n_size):
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + block_id * stride_mid_os
            + offs_d * stride_mid_od
        )
        tv = tl.load(Mid_O + offs_mid_o).to(tl.float32)
        
        offs_mid_lse = (
            cur_batch * stride_mid_les_b
            + cur_head * stride_mid_les_h
            + block_id * stride_mid_les_s
        )
        tlogic = tl.load(Mid_O_LogExpSum + offs_mid_lse).to(tl.float32)
        
        new_max = tl.maximum(max_logic, tlogic)
        old_scale = tl.exp(max_logic - new_max)
        acc = acc * old_scale
        sum_exp = sum_exp * old_scale
        
        exp_val = tl.exp(tlogic - new_max)
        acc += tv * exp_val
        sum_exp += exp_val
        max_logic = new_max
    
    acc = acc / sum_exp
    
    offs_out = cur_batch * stride_ob + cur_head * stride_oh + offs_d * stride_od
    tl.store(Out + offs_out, acc.to(Out.type.element_ty))

def flash_decode_stage2(
    Mid_O: torch.Tensor,
    Mid_O_LogExpSum: torch.Tensor,
    B_Seqlen: torch.Tensor,
    Out: torch.Tensor,
) -> torch.Tensor:
    batch, head_num = Mid_O.shape[0], Mid_O.shape[1]
    BLOCK_DMODEL = Mid_O.shape[-1]
    BLOCK_SEQ = Mid_O.shape[2]
    
    grid = (batch, head_num)
    
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
