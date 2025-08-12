
import torch
import triton
import triton.language as tl

@triton.jit
def cos_func(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a_ptr + offset, mask=mask).to(tl.float32)
    b_value = tl.cos(a_value)
    tl.store(b_ptr + offset, b_value, mask=mask)

def cos(a: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
    device_str = str(a.device).lower()
    assert any(k in device_str for k in ("cuda", "hip", "xpu")), "Expected AMD/ROCm or CUDA-like device"
    n_elements = a.numel()
    BLOCK_SIZE = triton.next_power_of_2(int(2 ** (int(torch.log2(torch.tensor(n_elements).float())) / 2)))
    if b is None:
        b = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    cos_func[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return b

##################################################################################################################################################



def test_cos_function():
    # Create test cases with various input sizes
    test_cases = {
        'test_case_1': torch.rand(1024, device='cuda') * 2 * math.pi,
        'test_case_2': torch.rand(2048, device='cuda') * 2 * math.pi,
        'test_case_3': torch.rand(4096, device='cuda') * 2 * math.pi,
        'test_case_4': torch.rand(8192, device='cuda') * 2 * math.pi
    }
    
    results = {}
    
    for case_name, input_tensor in test_cases.items():
        # Compute cosine using Triton
        B_triton = cos(input_tensor)
        results[case_name] = B_triton
    
    return results

# Run the test
result_gold = test_cos_function()
