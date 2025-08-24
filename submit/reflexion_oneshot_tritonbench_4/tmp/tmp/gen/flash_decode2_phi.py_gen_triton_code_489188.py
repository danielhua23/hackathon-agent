import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_out_b, stride_out_h, stride_out_d, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    blk_d = tl.arange(0, BLOCK_DMODEL)
    cur_seqlen = tl.load(B_Seqlen + cur_batch).to(tl.int32)
    block_n_size = (cur_seqlen + BLOCK_SEQ - 1) // BLOCK_SEQ
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for blk_idx in range(0, block_n_size):
        vid = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + blk_idx * stride_mid_os + blk_d * stride_mid_od
        lid = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + blk_idx * stride_mid_o_es
        tv = tl.load(Mid_O + vid).to(tl.float32)
        tlogic = tl.load(Mid_O_LogExpSum + lid).to(tl.float32)
        new_max = tl.maximum(max_logic, tlogic)
        scale = tl.exp(max_logic - new_max)
        acc *= scale
        sum_exp *= scale
        exp_val = tl.exp(tlogic - new_max)
        acc += exp_val * tv
        sum_exp += exp_val
        max_logic = new_max
    oid = cur_batch * stride_out_b + cur_head * stride_out_h + blk_d * stride_out_d
    tl.store(Out + oid, acc / sum_exp)

@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, Out, block_seq: int):
    B, H, seq_block, D = mid_out.shape
    assert D % 16 == 0 and D >= 16
    assert mid_out.dtype in (torch.float16, torch.float32)
    assert mid_out_logexpsum.dtype == mid_out.dtype
    assert B_Seqlen.shape[0] == B
    assert mid_out_logexpsum.shape == (B, H, seq_block)
    grid = (B, H)
    _fwd_kernel_flash_decode_stage2[grid](B_Seqlen, mid_out, mid_out_logexpsum, Out, mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2), Out.stride(0), Out.stride(1), Out.stride(2), BLOCK_SEQ=block_seq, BLOCK_DMODEL=D, num_warps=8, num_stages=2)

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
