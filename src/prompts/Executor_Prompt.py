from .Base import BasePrompt

class Executor_Prompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get_prompt(self, baseline_code, optimization_strategy) -> list:
        return [
            {
                "role": "system",
                "content": """You are a world-class expert in GPU kernel optimization, specializing in the Triton language for AMD GPUs.
You will be given the full source code of a Python file containing a baseline Triton kernel and a single, specific optimization strategy.
Your task is to apply **ONLY that single strategy** to the `@triton.jit` kernel function, while leaving the rest of the file completely untouched.
"""
            },
            {
                "role": "user",
                "content": f"""**Full Original Source Code:**
```python
{baseline_code}
```

**Optimization Strategy to Apply:**
"{optimization_strategy}"

**THE ABSOLUTE LAW (NON-NEGOTIABLE CORE DIRECTIVE):**

Your task is to **replicate the entire original Python file** with surgical precision. The ONLY part of the file you are allowed to modify is the internal implementation of the `@triton.jit` kernel function based on the strategy.

- **PRESERVE ALL OTHER CODE:** You **MUST** keep all other functions (wrapper functions, helpers, tests) and all other code (imports, comments) completely untouched.
- **NEVER MODIFY SIGNATURES:** You **MUST NOT** change the function signature of ANY function in the file.
- **YOUR SCOPE OF WORK:** Apply the optimization strategy *only* to the code inside the Triton kernel. The wrapper function's logic should generally remain the same unless the strategy requires a change (e.g., modifying grid computation).

**Final Verification Checklist:**
- [ ] Have I copied the *entire* original file content?
- [ ] Does every function signature in my output *exactly* match the original?
- [ ] Have I only modified the internal logic of the `@triton.jit` kernel according to the strategy?

**Now, generate the complete, correct, and surgically optimized Python file.**
"""
            }
        ]
