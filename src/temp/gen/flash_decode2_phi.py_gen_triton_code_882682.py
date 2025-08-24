
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
    stride_out_b,
    stride_out_h,
    stride_out_d,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head  = tl.program_id(1)
    offs_d    = tl.arange(0, BLOCK_DMODEL)

    seq_len   = tl.load(B_Seqlen + cur_batch)
    n_blocks  = tl.maximum((seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ, 1)

    acc       = tl.zeros([BLOCK_DMODEL], dtype=Mid_O.dtype.element_ty)
    max_logic = -float('inf')
    sum_exp   = 0.0

    for i in range(n_blocks):
        v  = tl.load(
            Mid_O + cur_batch * stride_mid_ob
                  + cur_head  * stride_mid_oh
                  + i         * stride_mid_os
                  + offs_d    * stride_mid_od
        )
        logsum = tl.load(
            Mid_O_LogExpSum + cur_batch * stride_mid_les_b
                            + cur_head  * stride_mid_les_h
                            + i         * stride_mid_les_s
        )
        new_max = tl.maximum(logsum, max_logic)
        scale   = tl.exp(max_logic - new_max)
        acc     = acc * scale + v * tl.exp(logsum - new_max)
        sum_exp = sum_exp * scale + tl.exp(logsum - new_max)
        max_logic = new_max

    acc = acc / sum_exp
    tl.store(
        Out + cur_batch * stride_out_b
            + cur_head  * stride_out_h
            + offs_d    * stride_out_d,
        acc
    )

def flash_decode_stage2(
    B_Seqlen: torch.Tensor,
    Mid_O: torch.Tensor,
    Mid_O_LogExpSum: torch.Tensor,
    Out: torch.Tensor,
    BLOCK_SEQ: int = 64,
    BLOCK_DMODEL: int = 128,
):
    B, H = Out.shape[0], Out.shape[1]
    grid = (B, H)
    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        Mid_O,
        Mid_O_LogExpSum,
        Out,
        Mid_O.stride(0),
        Mid_O.stride(1),
        Mid_O.stride(2),
        Mid_O.stride(3) if Mid_O.ndim == 4 else 1,
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
