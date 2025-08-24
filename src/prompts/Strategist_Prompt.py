from .Base import BasePrompt
import json

class Strategist_Prompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get_prompt(self, ps, analysis_json, baseline_code) -> list:
        # The analysis_json might be a string, ensure it's formatted nicely for the prompt.
        try:
            if isinstance(analysis_json, dict):
                analysis_str = json.dumps(analysis_json, indent=2)
            else:
                parsed_json = json.loads(analysis_json)
                analysis_str = json.dumps(parsed_json, indent=2)
        except (json.JSONDecodeError, TypeError):
            analysis_str = str(analysis_json) # Fallback to plain string representation

        return [
            {
                "role": "system",
                "content": """You are a world-class expert in GPU kernel optimization, specializing in the Triton language for AMD GPUs.
Your task is to analyze a baseline Triton kernel and propose a prioritized list of actionable, text-based optimization strategies. Your strategies must be grounded in specific code changes.
**DO NOT generate full code blocks.** Your output should be a numbered list of strategies, each with a brief hint about its implementation."""
            },
            {
                "role": "user",
                "content": f"""**Task Description:**
{ps.instruction}

**Task Analysis:**
```json
{analysis_str}
```

**Baseline Code:**
```python
{baseline_code}
```

**Your Task:**
Based on all the provided information, generate a numbered list of at least three specific, actionable optimization strategies to improve the performance of the 'Baseline Code'.

For each strategy, you **MUST** also briefly mention the key Triton functions (e.g., `tl.make_block_ptr`, `tl.load` with a mask) or the main code structure change (e.g., 'a new loop outside the main K-loop') that would be involved in its implementation. This provides a clear guide for the next agent.

For example:
1. Introduce `triton.autotune` to find the best block sizes. Implementation hint: Decorate the kernel with `@triton.autotune(configs=[...], key=[...])`.
2. Improve data loading for matrix 'B' by using 2D blocking. Implementation hint: Use `tl.make_block_ptr` and `tl.load` to handle block-based loading.

**Optimization Strategies:**
"""
            }
        ]
