import os
from tqdm import tqdm
from loguru import logger
import json
from dataclasses import asdict
from agents.Reflexion import Reflexion
from utils.utils import extract_function_signatures, clear_code, extract_function_calls, safe_force_correct_signature
from prompts import prompt_for_reflection
from memories.Memory import MemoryClassMeta
from models.Base import BaseModel
from agents.reflexion_oneshot import Reflexion_Oneshot
# --- Corrected Imports for Prompt Classes ---
from prompts.Analyst_Prompt import Analyst_Prompt
from prompts.Baseline_Prompt import Baseline_Prompt
from prompts.Strategist_Prompt import Strategist_Prompt
from prompts.Executor_Prompt import Executor_Prompt
from retrievers.retriever import BM25Retriever
from prompts import prompt_for_generation
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from utils.utils import clear_code
import os
import datetime
from tenacity import RetryError

# Inherit from Reflexion_Oneshot to gain all its functionality
class MultiAgentPipeline(Reflexion_Oneshot):
    """
    This agent augments the Reflexion framework by implementing a four-stage pipeline
    for the first iteration of code generation.
    """

    def __init__(self, model: BaseModel, dataset, corpus_path, mem_file=None):
        """
        Initializes the MultiAgentPipeline.
        It correctly calls the parent __init__ to set up all necessary components
        like BM25Retriever and memory, then adds its own pipeline-specific components.
        """
        # --- Corrected super().__init__() call ---
        # We must explicitly call the __init__ of our direct parent, Reflexion_Oneshot,
        # to ensure all its setup logic (retrievers, memory init) is executed.
        super().__init__(model, dataset, corpus_path, mem_file)

        # Initialize our custom prompt generators for the pipeline
        self.analyst_prompt_generator = Analyst_Prompt()
        self.baseline_prompt = Baseline_Prompt()
        self.strategist_prompt = Strategist_Prompt()
        self.executor_prompt = Executor_Prompt()
        
        # This dictionary will hold the state for our pipeline for each problem
        self.pipeline_states = {}
        
        # --- New: Setup for pipeline run outputs ---
        self.run_output_dir = os.path.join("/workspace", "pipeline_run_outputs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_output_dir, exist_ok=True)
        logger.info(f"Pipeline outputs will be saved to: {self.run_output_dir}")

        self._initialize_pipeline_states()

    def _save_pipeline_step(self, kernel_name: str, iteration: int, agent_name: str, step_type: str, content: str):
        """Helper function to save pipeline intermediate outputs."""
        try:
            kernel_dir = os.path.join(self.run_output_dir, kernel_name)
            os.makedirs(kernel_dir, exist_ok=True)
            file_name = f"iter_{iteration}_agent_{agent_name}_{step_type}.txt"
            file_path = os.path.join(kernel_dir, file_name)
            with open(file_path, "w") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to save pipeline step for {kernel_name}: {e}")


    def _initialize_pipeline_states(self):
        """
        Creates a parallel state management dictionary for our pipeline's needs,
        without modifying the original self.memories list.
        """
        logger.info("Initializing parallel pipeline states for each kernel...")
        for mem in self.memories:
            ps = mem.ps
            self.pipeline_states[ps.filename] = {
                "analysis": None,
                "baseline_code": None,
                "strategy_plan": [],
                "current_code": None,
                "best_code": None, # Will be updated based on evaluation
                "best_performance": float('-inf'),
                "current_strategy_idx": -1,
                "iteration": 0  # Initialize iteration counter
            }
        logger.info(f"{len(self.pipeline_states)} pipeline states initialized.")

    # The run method, and the pipeline-specific helper methods will be added next.
    # We will no longer touch memory_init, as super().__init__ handles it.

    def run(self, output_path=None, multi_thread=False, verbose=False, datalen=None, iteration_num=0, temperature=0):
        data_len = datalen if datalen else len(self.dataset)

        for i in range(iteration_num):
            logger.info(f"\n{'='*20} Iteration {i + 1}/{iteration_num} {'='*20}")

            # --- Generation Phase ---
            logger.info("Phase 1: Generating solutions...")
            # We iterate through the original self.memories list.
            for mem in tqdm(self.memories[:data_len], desc=f"Generation (Iter {i+1})"):
                if mem.pass_call:
                    continue
                
                if i == 0:
                    # --- Iteration 1: Use our full pipeline ---
                    self.run_full_pipeline(mem, temperature)
                    # After the first iteration, update the state
                    self.pipeline_states[mem.ps.filename]["iteration"] = 1
                else:
                    # --- Subsequent Iterations: Use our incremental optimizer ---
                    # This replaces the original generate_solution call
                    self.run_incremental_optimization(mem, temperature)
                    self.pipeline_states[mem.ps.filename]["iteration"] += 1

            # --- Evaluation Phase (adapted from original) ---
            logger.info("Phase 2: Evaluating solutions...")
            for mem in tqdm(self.memories[:data_len], desc=f"Evaluation (Iter {i+1})"):
                if mem.pass_call:
                    continue
                
                # The code to be tested is now in mem.ps.solution
                is_pass, err_msg = self.dataset.run_single_call(mem.ps)
                if not is_pass:
                    mem.err_msg = err_msg
                else:
                    mem.pass_call = True
                    mem.err_msg = None # Clear error message on pass
                    logger.info(f"  -> PASSED: {mem.ps.filename}")
            
            # --- Reflection Phase ---
            logger.info("Phase 3: Generating reflections and repairs for failures...")
            for mem in tqdm(self.memories[:data_len], desc=f"Reflection (Iter {i+1})"):
                if not mem.pass_call and mem.err_msg:
                    # This is where we implement our new two-step reflection process
                    self.diagnose_and_repair(mem, temperature)

            # --- File Writing (adapted from original) ---
            if output_path is not None:
                root, extension = os.path.splitext(output_path)
                iter_path = f"{root}_{i}{extension}"
                logger.info(f"Writing results for iteration {i+1} to {iter_path}")
                self.dataset.write_file(iter_path)
                    
    def run_full_pipeline(self, mem, temperature):
        """Runs the complete 4-agent pipeline and places the result in mem.ps.solution."""
        ps = mem.ps
        pipeline_state = self.pipeline_states[ps.filename]
        logger.info(f"  Running full pipeline for {ps.filename}...")

        try:
            # 1. Analyst
            analyst_prompt_obj = self.analyst_prompt_generator.get_prompt(ps) # Now returns a list
            self._save_pipeline_step(ps.filename, 1, "1_analyst", "input", json.dumps(analyst_prompt_obj, indent=2))
            analysis_str = self.model.generate(analyst_prompt_obj, temperature)
            pipeline_state["analysis"] = analysis_str
            self._save_pipeline_step(ps.filename, 1, "1_analyst", "output", analysis_str)

            # 2. Baseline
            baseline_prompt_obj = self.baseline_prompt.get_prompt(ps, analysis_str)
            self._save_pipeline_step(ps.filename, 1, "2_baseline", "input", json.dumps(baseline_prompt_obj, indent=2))
            baseline_code = self.model.generate(baseline_prompt_obj, temperature)
            baseline_code = clear_code(baseline_code)
            pipeline_state["baseline_code"] = baseline_code
            self._save_pipeline_step(ps.filename, 1, "2_baseline", "output", baseline_code)

            # 3. Strategist
            strategist_prompt_obj = self.strategist_prompt.get_prompt(ps, analysis_str, baseline_code)
            self._save_pipeline_step(ps.filename, 1, "3_strategist", "input", json.dumps(strategist_prompt_obj, indent=2))
            plan_str = self.model.generate(strategist_prompt_obj, temperature)
            strategies = self._parse_strategies(plan_str)
            pipeline_state["strategies"] = strategies
            pipeline_state["strategy_plan"] = plan_str
            self._save_pipeline_step(ps.filename, 1, "3_strategist", "output", plan_str)
            
            # 4. Executor (First Strategy)
            if strategies:
                first_strategy = strategies[0]
                pipeline_state["current_strategy_index"] = 0
                executor_prompt_obj = self.executor_prompt.get_prompt(baseline_code, first_strategy)
                self._save_pipeline_step(ps.filename, 1, "4_executor", "input", json.dumps(executor_prompt_obj, indent=2))
                executed_code = self.model.generate(executor_prompt_obj, temperature)
                executed_code = clear_code(executed_code)

                # --- NEW: Post-generation Signature Correction ---
                try:
                    # Infer function name from filename, with special handling for known variations
                    base_name = ps.filename.replace(".py", "")
                    name_map = {
                        'triton_matmul': 'matmul', 'matrix_vector_multip': 'mv',
                        'sin_kernel': 'call_kernel', 'matrix_transpose': 'wrapper',
                        'l2_norm_bwd': '_l2_norm_bwd'
                    }
                    func_name = name_map.get(base_name, base_name)

                    baseline_code = pipeline_state.get("baseline_code")

                    if baseline_code:
                        logger.info(f"  Applying SAFE signature correction for '{func_name}'...")
                        corrected_code = safe_force_correct_signature(baseline_code, executed_code, func_name)
                        
                        if corrected_code == executed_code:
                            logger.warning(f"  Signature correction for '{func_name}' was skipped (either safe or not needed).")
                        else:
                            logger.info(f"  Signature for '{func_name}' was successfully corrected.")

                        mem.ps.solution = corrected_code
                        pipeline_state["current_code"] = corrected_code
                        self._save_pipeline_step(ps.filename, 1, "5_corrected_executor", "output", corrected_code)
                    else:
                        logger.error("  Cannot perform signature correction: baseline_code not found.")
                        mem.ps.solution = executed_code
                        pipeline_state["current_code"] = executed_code
                        self._save_pipeline_step(ps.filename, 1, "4_executor", "output", executed_code)

                except Exception as e:
                    logger.error(f"  An unexpected error occurred during signature correction for {ps.filename}: {e}")
                    mem.ps.solution = executed_code
                    pipeline_state["current_code"] = executed_code
                    self._save_pipeline_step(ps.filename, 1, "4_executor", "output", executed_code)
                # --- End Correction ---

            else:
                # If no strategies, use the baseline code
                mem.ps.solution = baseline_code
                pipeline_state["current_code"] = baseline_code
        
        except RetryError as e:
            logger.error(f"  API call failed for {ps.filename} after multiple retries. Skipping this kernel.")
            error_message = f"API Error: The model server failed to respond.\n\nTraceback:\n{e}"
            self._save_pipeline_step(ps.filename, 1, "API_ERROR", "log", error_message)
            # Set a placeholder solution to indicate failure
            mem.ps.solution = "# API_ERROR: Model generation failed for this kernel."
            return

        return

    def run_incremental_optimization(self, mem, temperature):
        """Applies the next strategy and places the result in mem.ps.solution."""
        ps = mem.ps
        pipeline_state = self.pipeline_states[ps.filename]
        
        logger.info(f"Applying strategy {pipeline_state['current_strategy_index'] + 2}/{len(pipeline_state['strategies'])} for {ps.filename}...")

        # Move to the next strategy
        pipeline_state["current_strategy_index"] += 1
        
        # Get the next strategy
        strategy_index = pipeline_state["current_strategy_index"]
        
        if strategy_index < len(pipeline_state["strategies"]):
            strategy = pipeline_state["strategies"][strategy_index]
            
            # Get the last working code
            last_code = pipeline_state.get("current_code", pipeline_state.get("baseline_code"))
            if not last_code:
                logger.warning(f"No previous code found for {ps.filename}, cannot apply optimization. Skipping.")
                return

            try:
                # Use Executor to apply the new strategy
                executor_prompt_obj = self.executor_prompt.get_prompt(last_code, strategy)
                self._save_pipeline_step(ps.filename, pipeline_state['iteration'] + 1, "4_executor", "input", json.dumps(executor_prompt_obj, indent=2))
                executed_code = self.model.generate(executor_prompt_obj, temperature)
                executed_code = clear_code(executed_code)

                # --- NEW: Post-generation Signature Correction (Safe Version) ---
                try:
                    base_name = ps.filename.replace(".py", "")
                    name_map = {
                        'triton_matmul': 'matmul', 'matrix_vector_multip': 'mv',
                        'sin_kernel': 'call_kernel', 'matrix_transpose': 'wrapper',
                        'l2_norm_bwd': '_l2_norm_bwd'
                    }
                    func_name = name_map.get(base_name, base_name)
                    
                    baseline_code = pipeline_state.get("baseline_code")

                    if baseline_code:
                        logger.info(f"  Applying SAFE signature correction for '{func_name}' in iteration {pipeline_state['iteration'] + 1}...")
                        corrected_code = safe_force_correct_signature(baseline_code, executed_code, func_name)

                        if corrected_code == executed_code:
                            logger.warning(f"  Signature correction for '{func_name}' was skipped.")
                        else:
                            logger.info(f"  Signature for '{func_name}' was successfully corrected.")

                        mem.ps.solution = corrected_code
                        pipeline_state["current_code"] = corrected_code
                        self._save_pipeline_step(ps.filename, pipeline_state['iteration'] + 1, "5_corrected_executor", "output", corrected_code)
                    else:
                        logger.error("  Cannot perform signature correction: baseline_code not found.")
                        mem.ps.solution = executed_code
                        pipeline_state["current_code"] = executed_code
                        self._save_pipeline_step(ps.filename, pipeline_state['iteration'] + 1, "4_executor", "output", executed_code)
                
                except Exception as e:
                    logger.error(f"  An unexpected error occurred during signature correction for {ps.filename}: {e}")
                    mem.ps.solution = executed_code
                    pipeline_state["current_code"] = executed_code
                    self._save_pipeline_step(ps.filename, pipeline_state['iteration'] + 1, "4_executor", "output", executed_code)
                # --- End Correction ---

            except RetryError as e:
                logger.error(f"  API call failed during optimization for {ps.filename}. Skipping this optimization step.")
                error_message = f"API Error: The model server failed to respond during optimization.\n\nTraceback:\n{e}"
                self._save_pipeline_step(ps.filename, pipeline_state['iteration'] + 1, "API_ERROR", "log", error_message)
                # We don't have a new solution, so we just return and let the old one be evaluated
        else:
            logger.info(f"All strategies for {ps.filename} have been applied.")
            # If no more strategies, we let the default reflexion handle it
            pass

        return

    def _parse_strategies(self, plan_str: str) -> list[str]:
        """Helper to parse a numbered list of strategies from a string."""
        matches = re.findall(r"^\s*\d[\.\)-]\s*(.*)", plan_str, re.MULTILINE)
        if matches:
            return [match.strip() for match in matches]
        return [line.strip() for line in plan_str.split('\n') if line.strip()]

    # We are overriding the original generate_solution, but keeping generate_reflexion.
    def generate_solution(self, mem, temperature=0):
        # This is intentionally left blank because our pipeline's run_* methods
        # handle the logic that replaces this. The main `run` loop will no longer call this.
        pass

    def diagnose_and_repair(self, mem, temperature):
        """
        A new, two-step process for handling failures.
        1. Diagnose Failure and Create Plan: Use an expert prompt to create a high-level correction plan.
        2. Repair Code based on Plan: Use a second prompt to implement the plan and generate corrected code.
        """
        ps = mem.ps
        pipeline_state = self.pipeline_states[ps.filename]
        iteration = pipeline_state['iteration']

        # --- Step 1: Diagnose Failure and Create Plan ---
        logger.info(f"  Diagnosing failure for {ps.filename}...")
        
        # This is the expert diagnostician prompt string.
        diagnostician_prompt_template = """
You are an expert debugging assistant for Triton GPU kernels. You are given a Triton code snippet that has failed evaluation, along with its performance and error logs.
Your task is to create a concise, prioritized, and actionable correction plan.

**THE ABSOLUTE LAW:**
- Your plan MUST be a bulleted list starting with `-`.
- Do NOT propose fixing the function signature. A "Code Goalkeeper" has already corrected it. Focus on the kernel's internal logic and performance tuning.

**Input Information:**
1.  **Code with Issue:**
    ```python
    {code}
    ```
2.  **Evaluation Results:**
    - **Error Type:** "{error}"
    - **Error Trace/Log:** "{trace}"

**Your Task:** Based on the evaluation results, diagnose the primary failure mode and create a correction plan.

*   **If `Error Type` is `Runtime Error`:** Analyze the traceback. Pinpoint the likely cause (e.g., shape mismatch in `tl.dot`, memory access error).
*   **If `Error Type` is `Correctness Error`:** This is a logic error. Suspect issues in pointer arithmetic, accumulator updates, or incorrect masking.
*   **If `Error Type` is `Success` but performance is low:** This is a tuning problem. Suggest changes to block sizes, num_warps, etc.

**Correction Plan:**
"""

        diagnose_prompt = diagnostician_prompt_template.format(
            code=ps.solution,
            error=mem.err_msg.get('error_type', 'Unknown'),
            trace=mem.err_msg.get('error_log', 'No log available'),
            instruction=ps.instruction
        )
        
        correction_plan = self.model.generate([{"role": "user", "content": diagnose_prompt}], temperature)
        pipeline_state["correction_plan"] = correction_plan
        self._save_pipeline_step(ps.filename, iteration, "6_correction_plan", "output", correction_plan)
        logger.info(f"  Correction plan for {ps.filename} generated.")

        # --- Step 2: Repair Code based on Plan ---
        logger.info(f"  Repairing code for {ps.filename} based on the new plan...")
        from prompts.prompt_for_repair import prompt as repair_prompt_template
        
        repair_prompt = repair_prompt_template.format(
            solution=ps.solution,
            test_result=mem.err_msg.get('error_log', 'No log available'),
            reflection=correction_plan  # Feed the plan into the repair prompt
        )

        repaired_code = self.model.generate([{"role": "user", "content": repair_prompt}], temperature)
        repaired_code = clear_code(repaired_code)

        # --- Step 3: Apply Goalkeeper to the repaired code ---
        logger.info(f"  Applying Goalkeeper to the repaired code for {ps.filename}...")
        
        base_name = ps.filename.replace(".py", "")
        name_map = {
            'triton_matmul': 'matmul', 'matrix_vector_multip': 'mv',
            'sin_kernel': 'call_kernel', 'matrix_transpose': 'wrapper',
            'l2_norm_bwd': '_l2_norm_bwd'
        }
        func_name = name_map.get(base_name, base_name)
        baseline_code = pipeline_state.get("baseline_code")

        if baseline_code:
            final_code = safe_force_correct_signature(baseline_code, repaired_code, func_name)
        else:
            logger.warning(f"  Baseline code not found for {ps.filename}, Goalkeeper skipped on repaired code.")
            final_code = repaired_code

        # Update memory and state with the new, repaired, and corrected solution
        mem.ps.solution = final_code
        pipeline_state["current_code"] = final_code
        self._save_pipeline_step(ps.filename, iteration, "7_repaired_code", "output", final_code)
        logger.info(f"  Code for {ps.filename} has been repaired and corrected for the next iteration.")

    # We keep the original generate_reflexion, but it will no longer be called by our main loop.
    def generate_reflexion(self, mem, temperature):
        if mem.pass_call:
            return
        reflect_txt = prompt_for_reflection.prompt.format(
            problem=mem.ps.instruction,
            solution=mem.ps.solution,
            test_result=mem.err_msg
        )
        reflect_msg = [
            {
                "role": "user",
                "content": reflect_txt
            }
        ]
        mem.reflection = self.model.generate(reflect_msg, temperature=temperature)