import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_esb, stride_mid_o_esh, stride_mid_o_ess, stride_out_b, stride_out_h, stride_out_d, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    seq_len = tl.load(B_Seqlen + cur_batch)
    block_n_size = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    sum_exp = 0.0
    max_logic = float('-inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for block_id in range(0, block_n_size):
        tv = tl.load(Mid_O + cur_batch * stride_mid_ob + cur_head * stride_mid_oh + block_id * stride_mid_os + offs_d * stride_mid_od)
        tlogic = tl.load(Mid_O_LogExpSum + cur_batch * stride_mid_o_esb + cur_head * stride_mid_o_esh + block_id * stride_mid_o_ess)
        new_max = tl.maximum(max_logic, tlogic)
        scale = tl.exp(max_logic - new_max)
        acc = acc * scale
        sum_exp = sum_exp * scale
        exp_logic = tl.exp(tlogic - new_max)
        acc = acc + tv.to(tl.float32) * exp_logic
        sum_exp = sum_exp + exp_logic
        max_logic = new_max
    acc = acc / sum_exp
    tl.store(Out + cur_batch * stride_out_b + cur_head * stride_out_h + offs_d * stride_out_d, acc.to(Out.dtype.element_ty))

@torch.no_grad()
def flash_decode_stage2(B_Seqlen, mid_out, mid_out_logexpsum, output, block_seq):
    BLOCK_DMODEL = mid_out.size(-1)
    batch = B_Seqlen.shape[0]
    head_num = mid_out.shape[1]
    grid = (batch, head_num)
    _fwd_kernel_flash_decode_stage2[grid](B_Seqlen, mid_out, mid_out_logexpsum, output, mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2), output.stride(0), output.stride(1), output.stride(2), BLOCK_SEQ=block_seq, BLOCK_DMODEL=BLOCK_DMODEL, num_warps=4, num_stages=2)

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
