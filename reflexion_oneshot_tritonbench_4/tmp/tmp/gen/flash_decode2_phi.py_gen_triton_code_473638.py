import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_obs, stride_oh, stride_od, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    seq_len = tl.load(B_Seqlen + cur_batch)
    block_n_size = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d * stride_mid_od
    offs_l = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_id in range(0, block_n_size):
        tv = tl.load(Mid_O + offs_v + block_id * stride_mid_os)
        l_cur = tl.load(Mid_O_LogExpSum + offs_l + block_id * stride_mid_o_es)
        new_max = tl.maximum(max_logic, l_cur)
        old_scale = tl.exp(max_logic - new_max)
        acc *= old_scale
        sum_exp *= old_scale
        cur_exp = tl.exp(l_cur - new_max)
        acc += tv * cur_exp
        sum_exp += cur_exp
        max_logic = new_max
    tl.store(Out + cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od, (acc / sum_exp).to(Out.dtype.element_ty))

@torch.no_grad()
def flash_decode_stage2(mid_out: torch.Tensor, mid_out_logexpsum: torch.Tensor, B_Seqlen: torch.Tensor, out: torch.Tensor, block_seq: int, BLOCK_DMODEL: int=None) -> None:
    if BLOCK_DMODEL is None:
        BLOCK_DMODEL = out.shape[-1]
    batch, head_num = out.shape[:2]
    grid = (batch, head_num)
    _fwd_kernel_flash_decode_stage2[grid](B_Seqlen, mid_out, mid_out_logexpsum, out, mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3), mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1), mid_out_logexpsum.stride(2), out.stride(0), out.stride(1), out.stride(2), BLOCK_SEQ=block_seq, BLOCK_DMODEL=BLOCK_DMODEL, num_warps=4, num_stages=2)

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
