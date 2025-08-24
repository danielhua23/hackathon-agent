from .Base import BasePrompt
import json

class Baseline_Prompt(BasePrompt):
    def __init__(self):
        super().__init__()
    
    def get_prompt(self, ps, analysis_json) -> list:
        # The analysis_json might be a string, ensure it's formatted nicely for the prompt.
        try:
            # If it's a dict, dump it to a string
            if isinstance(analysis_json, dict):
                analysis_str = json.dumps(analysis_json, indent=2)
            else:
                # If it's already a string, try to parse and re-format it to be safe
                parsed_json = json.loads(analysis_json)
                analysis_str = json.dumps(parsed_json, indent=2)
        except (json.JSONDecodeError, TypeError):
            analysis_str = str(analysis_json) # Fallback to plain string representation

        return [
            {
                "role": "system",
                "content": """You are an expert Python programmer specializing in Triton kernels for AMD GPUs (ROCm).
Your task is to write a simple, correct, and easy-to-read Triton kernel.
**DO NOT focus on performance optimization at this stage.** Your only goal is to generate a functionally correct baseline implementation.
"""
            },
            {
                "role": "user",
                "content": f"""**Task Description:**
{ps.instruction}

**Task Analysis:**
```json
{analysis_str}
```

**Test Code (for context on how the function will be called):**
```python
{ps.test_code}
```

**Output Requirements:**
1.  **Signature Matching:** Your generated function's signature (name and parameters) **MUST EXACTLY MATCH** how it is called in the provided `test_code`. Analyze the test code carefully to determine the correct signature.
2.  **AMD Compatibility:** Ensure the code is compatible with AMD GPUs and ROCm. Do not use CUDA-specific features.
3.  **Complete & Simple Code:** Generate a single, complete Python code block. The logic should be as straightforward as possible.
4.  **Use `tl.dot` for Matrix Multiplication:** For matrix multiplication operations (matrix-matrix or matrix-vector), you **MUST** use the `tl.dot` instruction. Do not implement it manually with element-wise multiplication and summation.
5.  **Basic Triton Kernel:** Implement the core logic in a `@triton.jit` function.
6.  **Imports:** Include necessary imports like `torch`, `triton`, and `triton.language as tl`.
7.  **No Advanced Optimization:**
    *   **DO NOT** use `triton.autotune`.
    *   **DO NOT** implement complex tiling or shared memory strategies unless absolutely necessary for correctness.
"""
            }
        ]
