
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,               # [head_num, seq_blocks, BLOCK_DMODEL]
    Mid_O_LogExpSum,     # [head_num, seq_blocks]
    Out,                 # [head_num, BLOCK_DMODEL]
    stride_mid_oh,       # stride(head_num)
    stride_mid_ob,       # stride(seq_blocks)
    stride_mid_od,       # stride(BLOCK_DMODEL)
    stride_mid_o_lseh,   # stride(head)
    stride_mid_o_lseb,   # stride(seq_blocks)
    stride_oh,           # stride(head_num)
    stride_od,           # stride(BLOCK_DMODEL)
    B_START_ID,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    for block_id in range(0, block_n_size):
        ptr_tv = (
            Mid_O
            + (cur_head * stride_mid_oh)
            + (block_id * stride_mid_ob)
            + offs_d * stride_mid_od
        )

        tv = tl.load(ptr_tv)
        ptr_tlogic = (
            Mid_O_LogExpSum
            + cur_head * stride_mid_o_lseh
            + block_id * stride_mid_o_lseb
        )
        tlogic = tl.load(ptr_tlogic)

        max_prev = max_logic
        max_logic = tl.maximum(max_prev, tlogic)
        sum_exp = sum_exp * tl.exp(max_prev - max_logic) + tl.exp(tlogic - max_logic)
        acc = acc * tl.exp(max_prev - max_logic) + tv * tl.exp(tlogic - max_logic)

    result = acc / (sum_exp + 1e-6)

    ptr_out = Out + cur_head * stride_oh + offs_d * stride_od
    tl.store(ptr_out, result.cast(ptr_out.dtype.element_ty))


def flash_decode_stage2(
    B_Seqlen: torch.Tensor,
    Mid_O: torch.Tensor,
    Mid_O_LogExpSum: torch.Tensor,
    Out: torch.Tensor,
    B_START_ID: int = 0,
    BLOCK_SEQ: int = 64,
):
    head_num, seq_blocks, BLOCK_DMODEL = Mid_O.shape
    batch_size = B_Seqlen.shape[0]

    triton_grid = (batch_size, head_num)

    _fwd_kernel_flash_decode_stage2[
        triton_grid
    ](
        B_Seqlen,
        Mid_O,
        Mid_O_LogExpSum,
        Out,
        Mid_O.stride(0),
        Mid_O.stride(1),
        Mid_O.stride(2),
        Mid_O_LogExpSum.stride(0),
        Mid_O_LogExpSum.stride(1),
        Out.stride(0),
        Out.stride(1),
        B_START_ID,
        BLOCK_SEQ=BLOCK_SEQ,
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
