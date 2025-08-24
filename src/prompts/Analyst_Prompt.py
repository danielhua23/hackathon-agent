from .Base import BasePrompt

class Analyst_Prompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get_prompt(self, ps) -> list:
        # Reverting to the original and correct list-based format.
        return [
            {
                "role": "system",
                "content": "You are a top-tier Triton code analysis expert. Please read the following task description and strictly summarize its core computational type, key performance-affecting parameters, and at least three potential optimization directions in a JSON format. Do not add any explanatory text outside of the JSON structure."
            },
            {
                "role": "user",
                "content": f"""Task Description:
{ps.instruction}

Please provide the analysis in the following JSON structure:
{{
  "type": "<e.g., matmul, element-wise, memory-bound>",
  "key_parameters": ["<param1>", "<param2>", ...],
  "optimization_hints": ["<hint1>", "<hint2>", "<hint3>"]
}}"""
            }
        ]
