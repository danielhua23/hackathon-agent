
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M,
    Out,
    matrix_stridex,
    matrix_stridey,
    out_stridex,
    out_stridey,
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = tl.arange(0, SIZE_M)
    offs_n = tl.arange(0, D_HEAD)

    matrix_ptr = M + offs_n[None, :] * matrix_stridex + offs_m[:, None] * matrix_stridey
    out_ptr    = Out + offs_n[None, :] * out_stridey + offs_m[:, None] * out_stridex

    x = tl.load(matrix_ptr)
    tl.store(out_ptr, x)


def wrapper(SIZE_M: int, D_HEAD: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = torch.randn((SIZE_M, D_HEAD), dtype=torch.float16, device=device)
    out = torch.zeros((D_HEAD, SIZE_M), dtype=torch.float16, device=device)

    grid = (D_HEAD, SIZE_M)
    kernel[grid](
        matrix,
        out,
        matrix.stride(1),
        matrix.stride(0),
        out.stride(1),
        out.stride(0),
        SIZE_M=SIZE_M,
        D_HEAD=D_HEAD,
    )

    return out
