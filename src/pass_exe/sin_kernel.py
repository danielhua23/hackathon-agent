
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.math.sin(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def call_kernel(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel_function[grid](
        x, output, n_elements,
        BLOCK_SIZE=1024,
    )
    return output
