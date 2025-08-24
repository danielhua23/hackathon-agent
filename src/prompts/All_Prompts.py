from .Base import BasePrompt
import json

class Baseline_Prompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get_prompt_str(self, ps, analysis_json) -> str:
        # This method is for compatibility with the new _save_pipeline_step which expects a string
        prompt_list = self.get_prompt(ps, analysis_json)
        # We'll just serialize the whole prompt structure for saving
        return json.dumps(prompt_list, indent=2)

    def get_prompt(self, ps, analysis_json) -> list:
        # The analysis_json might be a string, ensure it's formatted nicely for the prompt.
        try:
            if isinstance(analysis_json, dict):
                return analysis_json
            elif isinstance(analysis_json, str):
                return json.loads(analysis_json)
            else:
                return []
        except json.JSONDecodeError:
            return []

from .Base import BasePrompt

class Analyst_Prompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get_prompt_str(self, ps) -> str:
        # This method is for compatibility with the new _save_pipeline_step which expects a string
        prompt_list = self.get_prompt(ps)
        return json.dumps(prompt_list, indent=2)

    def get_prompt(self, ps) -> list:
        return [
            {
                "role": "user",
                "content": "You are an analyst. Your task is to analyze the provided data and provide insights. Please provide a detailed analysis of the data, including any patterns, trends, or anomalies you observe. Do not make any assumptions or guesses; only provide factual information based on the data."
            }
        ]

from .Base import BasePrompt

class Executor_Prompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get_prompt_str(self, baseline_code, optimization_strategy) -> str:
        # This method is for compatibility with the new _save_pipeline_step which expects a string
        prompt_list = self.get_prompt(baseline_code, optimization_strategy)
        return json.dumps(prompt_list, indent=2)

    def get_prompt(self, baseline_code, optimization_strategy) -> list:
        return [
            {
                "role": "user",
                "content": "You are an executor. Your task is to execute the provided baseline code and optimize it based on the optimization strategy. Please provide a step-by-step explanation of the execution process and the optimization steps taken. Do not make any assumptions or guesses; only provide factual information based on the code and strategy."
            }
        ]

from .Base import BasePrompt
import json

class Strategist_Prompt(BasePrompt):
    def __init__(self):
        super().__init__()

    def get_prompt_str(self, ps, analysis_json, baseline_code) -> str:
        # This method is for compatibility with the new _save_pipeline_step which expects a string
        prompt_list = self.get_prompt(ps, analysis_json, baseline_code)
        return json.dumps(prompt_list, indent=2)

    def get_prompt(self, ps, analysis_json, baseline_code) -> list:
        try:
            if isinstance(analysis_json, dict):
                return analysis_json
            elif isinstance(analysis_json, str):
                return json.loads(analysis_json)
            else:
                return []
        except json.JSONDecodeError:
            return []
