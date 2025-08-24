import re
import json
import ast

def extract_function_signatures(code):
    function_defs = []
    pattern = r'def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)'
    matches = re.finditer(pattern, code)
    
    for match in matches:
        func_name = match.group(1)
        params = match.group(2)
        function_defs.append(f"def {func_name}({params})")
    
    return function_defs

def clear_code(code):
    if  "```python" in code:
        code = code.split("```python")[-1].replace("<|im_end|>", "").replace("<|EOT|>", "")
    if "```" in code:
        code = code.split("```")[0]
    return code

def extract_function_calls(code):
    calls = []
    pattern = r'([a-zA-Z0-9_]+)\s*\(([^)]*)\)'
    matches = re.finditer(pattern, code)
    
    for match in matches:
        func_name = match.group(1)
        args = match.group(2)
        calls.append(f"{func_name}({args})")
    
    return calls

def clear_json(response):
    if type(response) is dict:
        return response
    elif type(response) is not str:
        response = str(response)
    try:
        response = response.replace("\n", " ")
        response = re.search('({.+})', response).group(0)
        response = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", response)
        result = ast.literal_eval(response)
    except (SyntaxError, NameError, AttributeError):
        return "ERR_SYNTAX"
    return result

def safe_force_correct_signature(agent2_baseline_code: str, agent4_optimized_code: str, func_name: str) -> str:
    """
    Safely corrects the signature of a function in Agent 4's optimized code to match
    the signature from Agent 2's baseline code.

    It performs a safety check to ensure the optimized function's parameters are a
    superset of the baseline's parameters before performing the replacement.

    Returns the corrected code, or the original optimized code if correction is not possible or safe.
    """
    try:
        # Step A: Extract "golden" info from Agent 2's baseline code
        baseline_tree = ast.parse(agent2_baseline_code)
        golden_node = None
        for node in ast.walk(baseline_tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                golden_node = node
                break
        
        if not golden_node:
            # print(f"Warning: Golden function '{func_name}' not found in baseline code.")
            return agent4_optimized_code

        golden_signature = ast.get_source_segment(agent2_baseline_code, golden_node).split(':\\n')[0]
        golden_params = {arg.arg for arg in golden_node.args.args}

        # Step B: Extract "to-check" info from Agent 4's optimized code
        optimized_tree = ast.parse(agent4_optimized_code)
        optimized_node = None
        for node in ast.walk(optimized_tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                optimized_node = node
                break
        
        if not optimized_node:
            # print(f"Warning: Function '{func_name}' not found in optimized code.")
            return agent4_optimized_code

        optimized_params = {arg.arg for arg in optimized_node.args.args}

        # Step C: Safety Check
        if not golden_params.issubset(optimized_params):
            # print(f"Safety check failed: Optimized params {optimized_params} is not a superset of golden params {golden_params}.")
            return agent4_optimized_code # Return original code if unsafe

        # Step D: Execute Safe String Replacement
        lines = agent4_optimized_code.splitlines()
        start_line_idx = optimized_node.lineno - 1
        end_line_idx = optimized_node.body[0].lineno - 1
        lines[start_line_idx : end_line_idx] = [golden_signature]

        return "\n".join(lines)

    except (SyntaxError, IndexError) as e:
        # print(f"Error processing code for signature correction: {e}")
        return agent4_optimized_code # Return original code on parsing error

# Keep the old function for now, might be useful for other things, but we'll use the safe one.
def force_correct_signature(generated_code: str, golden_signature: str, func_name: str) -> str:
    """
    Finds a function by name in the generated code using an AST parser and
    replaces its signature with the provided golden standard signature.
    This is a robust way to enforce signature correctness against LLM hallucinations.
    """
    try:
        tree = ast.parse(generated_code)
        
        target_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                target_node = node
                break

        if target_node:
            lines = generated_code.splitlines()
            
            # AST line numbers are 1-based, list indices are 0-based
            start_line_idx = target_node.lineno - 1
            
            # Find the end of the signature (start of the function body)
            # The end line index is the line number of the first statement in the function body, minus one.
            end_line_idx = target_node.body[0].lineno - 1

            # Replace all lines from the function definition start to just before the body
            # with the single, correct golden signature line.
            # This handles multi-line signatures and docstrings incorrectly placed before the body.
            lines[start_line_idx : end_line_idx] = [golden_signature.strip()]
            
            return "\\n".join(lines)
        else:
            # If the target function isn't found, return the original code to avoid breaking things.
            # A warning could be logged here in a real application.
            # print(f"Warning: Function '{func_name}' not found in generated code.")
            return generated_code

    except SyntaxError as e:
        # If the generated code is not valid Python, we can't parse it.
        # Return the original code and let the evaluation process handle the syntax error.
        # print(f"Error parsing generated code: {e}")
        return generated_code