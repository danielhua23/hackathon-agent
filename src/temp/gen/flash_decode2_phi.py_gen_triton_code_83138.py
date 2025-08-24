
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    Out,
    stride_bseqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_olesb,
    stride_mid_olesh,
    stride_mid_oles,
    stride_oub,
    stride_ouh,
    stride_oud,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_head = tl.program_id(1)
    cur_batch = tl.program_id(0)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    sum_exp = 0.0
    max_logic = float("-inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    seq_len = tl.load(B_Seqlen + cur_batch * stride_bseqlen)
    block_n_size = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    for block_n in range(block_n_size):
        tv = tl.load(Mid_O + cur_batch * stride_mid_ob + cur_head * stride_mid_oh +
                     block_n * stride_mid_os + offs_d * stride_mid_od)
        tlogic = tl.load(Mid_O_LogExpSum + cur_batch * stride_mid_olesb +
                         cur_head * stride_mid_olesh + block_n * stride_mid_oles)

        new_max_logic = tl.maximum(max_logic, tlogic)
        old_scale = tl.exp(max_logic - new_max_logic)
        new_scale = tl.exp(tlogic - new_max_logic)

        acc = acc * old_scale + tv * new_scale
        sum_exp = sum_exp * old_scale + new_scale
        max_logic = new_max_logic

    acc = acc / sum_exp
    tl.store(Out + cur_batch * stride_oub + cur_head * stride_ouh + offs_d * stride_oud, acc)


def flash_decode_stage2(
    Mid_O: torch.Tensor,
    Mid_O_LogExpSum: torch.Tensor,
    B_Seqlen: torch.Tensor,
    Out: torch.Tensor,
    block_seq: int
):
    batch = B_Seqlen.shape[0]
    head_num = Mid_O.shape[1]
    assert Mid_O_LogExpSum.shape[1] == head_num

    BLOCK_SEQ = block_seq
    BLOCK_DMODEL = Mid_O.shape[3]

    _fwd_kernel_flash_decode_stage2[(batch, head_num)](
        B_Seqlen,
        Mid_O,
        Mid_O_LogExpSum,
        Out,
        B_Seqlen.stride(0),
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
        num_stages=2
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
