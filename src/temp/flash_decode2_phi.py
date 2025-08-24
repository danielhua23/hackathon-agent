

import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,        # int32[B]          number of valid tokens per batch
    Mid_O,           # float[B, H, Sb, D] partial outputs per block
    Mid_O_LogExpSum, # float[B, H, Sb]    log-sum-exp of the logits per block
    Out,             # float[B, H, D]     final output
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_ls_oh, stride_ls_os,
    stride_out_ob, stride_out_oh, stride_out_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    pid_b = tl.program_id(0)  # batch dimension
    pid_h = tl.program_id(1)  # head dimension

    # ------------------------------------------------------------------
    # Offsets for the first element of the current (batch, head) slice.
    # ------------------------------------------------------------------
    mid_o_ptr = Mid_O + pid_b * stride_mid_ob + pid_h * stride_mid_oh
    mid_ls_ptr = Mid_O_LogExpSum + pid_b * stride_mid_ob + pid_h * stride_mid_oh

    # Load actual sequence length for this batch
    seq_len = tl.load(B_Seqlen + pid_b)
    block_n_size = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ  # ceil division

    if block_n_size == 0:
        return

    # Init per-thread accumulators
    max_logic = -float('inf')
    sum_exp = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for bn in range(0, block_n_size):
        # Load the log-sum-exp for this block
        block_lse = tl.load(mid_ls_ptr + bn * stride_ls_os)

        # Update running range reduction
        new_max = tl.maximum(max_logic, block_lse)
        scale = tl.exp(max_logic - new_max)
        new_scale = tl.exp(block_lse - new_max)

        # Scale prior accumulation
        sum_exp = sum_exp * scale + new_scale
        acc = acc * scale

        # Load partial output
        tv = tl.load(mid_o_ptr + bn * stride_mid_os +
                     tl.arange(0, BLOCK_DMODEL)[None, :] * stride_mid_od)

        acc = acc + tv * new_scale  # broadcasted multiply over BLOCK_DMODEL
        max_logic = new_max

    # Normalize
    inv_sum_exp = 1.0 / sum_exp
    acc = acc * inv_sum_exp

    # Store final result
    out_ptr = Out + pid_b * stride_out_ob + pid_h * stride_out_oh
    tl.store(out_ptr + tl.arange(0, BLOCK_DMODEL) * stride_out_od,
             acc)


def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, Out, block_seq: int):
    """
    Top-level wrapper that launches the Triton kernel above.
    mid_out: shape (B, H, Sb, D)
    mid_out_logexpsum: shape (B, H, Sb)
    B_Seqlen: shape (B,) dtype int32
    Out: shape (B, H, D)
    block_seq: integer constant equal to BLOCK_SEQ on which the kernel was built.
    """
    B, H, Sb, D = mid_out.shape

    # Determine launch grid
    grid = (B, H)

    _fwd_kernel_flash_decode_stage2[grid](
        B_Seqlen,
        mid_out,
        mid_out_logexpsum,
        Out,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logexpsum.stride(0), mid_out_logexpsum.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=D,
        num_warps=4
    )


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
