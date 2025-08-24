import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_obs, stride_oh, stride_od, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr, SEQ_BLK_MAX: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    seq_len = tl.load(B_Seqlen + pid_b).to(tl.int32)
    block_n_size = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    max_logic = tl.full([], float('-inf'), dtype=tl.float32)
    sum_exp = tl.full([], 0.0, dtype=tl.float32)
    for blk in tl.static_range(SEQ_BLK_MAX):
        valid = blk < block_n_size
        ptr_mid = Mid_O + (pid_b * stride_mid_ob + pid_h * stride_mid_oh + blk * stride_mid_os + offs_d * stride_mid_od)
        ptr_logic = Mid_O_LogExpSum + (pid_b * stride_mid_o_eb + pid_h * stride_mid_o_eh + blk * stride_mid_o_es)
        tv = tl.load(ptr_mid, mask=valid & (offs_d < BLOCK_DMODEL), other=0.0)
        tlogic = tl.load(ptr_logic, mask=valid, other=float('-inf'))
        new_max = tl.maximum(max_logic, tlogic)
        exp_old = tl.exp(max_logic - new_max)
        exp_new = tl.exp(tlogic - new_max)
        acc = acc * exp_old + tv * exp_new
        sum_exp = sum_exp * exp_old + exp_new
        max_logic = new_max
    final = tl.where(block_n_size > 0, acc / (sum_exp + 1e-06), 0.0)
    ptr_out = Out + pid_b * stride_obs + pid_h * stride_oh + offs_d * stride_od
    tl.store(ptr_out, final.to(Out.type.element_ty), mask=offs_d < BLOCK_DMODEL)

@triton.autotune(configs=[triton.Config({'BLOCK_SEQ': 32, 'BLOCK_DMODEL': 64, 'SEQ_BLK_MAX': 64}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SEQ': 64, 'BLOCK_DMODEL': 128, 'SEQ_BLK_MAX': 128}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SEQ': 64, 'BLOCK_DMODEL': 256, 'SEQ_BLK_MAX': 512}, num_stages=2, num_warps=8)], key=['head_dim', 'max_seq_blocks'])
@triton.jit
def _fwd_kernel_flash_decode_stage2_tuned(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_obs, stride_oh, stride_od, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr, SEQ_BLK_MAX: tl.constexpr):
    _fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_obs, stride_oh, stride_od, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL, SEQ_BLK_MAX=SEQ_BLK_MAX)

def flash_decode_stage2(Mid_O, Mid_O_LogExpSum, B_Seqlen, Out, max_seqlen):
    """
    Mid_O: [batch, heads, seq_blocks, head_dim]
    Mid_O_LogExpSum: [batch, heads, seq_blocks]
    B_Seqlen: [batch] (torch.int32)
    Out: [batch, heads, head_dim] (output, must exist and be correct dtype)
    max_seqlen: int
    """
    assert B_Seqlen.dim() == 1
    assert Mid_O.dim() == 4
    assert Mid_O_LogExpSum.dim() == 3
    b, h, seq_blocks, head_dim = Mid_O.size()
    assert Mid_O_LogExpSum.size() == (b, h, seq_blocks)
    assert Out.size() == (b, h, head_dim)
    seq_blk_max = triton.cdiv(seq_blocks, 1)
    head_dim_pow2 = max(64, triton.next_power_of_2(head_dim))
    grid = (b, h)
    _fwd_kernel_flash_decode_stage2_tuned[grid](B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, Mid_O.stride(0), Mid_O.stride(1), Mid_O.stride(2), Mid_O.stride(3), Mid_O_LogExpSum.stride(0), Mid_O_LogExpSum.stride(1), Mid_O_LogExpSum.stride(2), Out.stride(0), Out.stride(1), Out.stride(2), head_dim=head_dim, max_seq_blocks=seq_blk_max)
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
