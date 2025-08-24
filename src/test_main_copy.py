
import os
from agents.reflexion_oneshot import Reflexion_Oneshot
from models.KimiK2 import KimiK2Model
from dataloaders.TritonBench import TritonBench
from args_config import load_config
import json
from prompts.Baseline_Prompt import Baseline_Prompt # Import the new Baseline_Prompt


# --- Pre-defined inputs from Agent 1 ---
# This is the JSON output we got from our previous test for triton_matmul.py
MATMUL_ANALYSIS_JSON = {
  "type": "matmul",
  "key_parameters": [
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
    "num_stages", "num_warps", "K", "M", "N"
  ],
  "optimization_hints": [
    "Tune BLOCK_SIZE_* to maximize occupancy and shared-memory utilization",
    "Exploit tensor-core-accelerated tl.dot with fp16/bf16 inputs and appropriate block sizes",
    "Pipeline loads with num_stages>1 to overlap global memory transfers with compute"
  ]
}

# This is the JSON output we got for matrix_transpose.py
TRANSPOSE_ANALYSIS_JSON = {
  "type": "memory-bound transpose",
  "key_parameters": [
    "SIZE_M", "D_HEAD", "matrix_stridex", "matrix_stridey",
    "out_stridex", "out_stridey"
  ],
  "optimization_hints": [
    "use blocked 2-D thread indexing with tile-based shared memory to enable coalesced reads and writes",
    "vectorize loads/stores via tl.load/store with appropriate masks and vector widths (e.g., 4 or 8 fp16 elements)",
    "tune block size and tile dimensions to maximize occupancy and L2 cache hit rate while minimizing bank conflicts"
  ]
}

AGENT1_OUTPUTS = {
    'triton_matmul.py': MATMUL_ANALYSIS_JSON,
    'matrix_transpose.py': TRANSPOSE_ANALYSIS_JSON
}
# -----------------------------------------


def run_agent_2_test(model, dataset):
    print("\n" + "="*80)
    print(">>> RUNNING TEST FOR AGENT 2: BASELINE IMPLEMENTER <<<")
    print("="*80 + "\n")

    baseline_prompt_generator = Baseline_Prompt()
    kernels_to_test = ['triton_matmul.py', 'matrix_transpose.py']

    print(f"Starting baseline code generation for {len(kernels_to_test)} kernels.")
    print("-" * 80)

    for problem in dataset.problem_states:
        if hasattr(problem, 'filename') and problem.filename in kernels_to_test:
            print(f"\n>>> Generating Baseline for Kernel: {problem.filename}")
            
            # Get the corresponding analysis from our pre-defined dictionary
            analysis_json = AGENT1_OUTPUTS[problem.filename]
            print("\n--- Input from Agent 1 (Analysis JSON) ---")
            print(json.dumps(analysis_json, indent=2))
            
            prompt_messages = baseline_prompt_generator.get_prompt(ps=problem, analysis_json=analysis_json)
            
            print("\n...Calling KimiK2 API to generate baseline code...")
            try:
                baseline_code = model.generate(messages=prompt_messages, temperature=0.0)
                print("\n--- LLM Raw Output (Baseline Code) ---")
                print(baseline_code)

            except Exception as e:
                import traceback
                print(f"\n[ERROR] Test failed for kernel {problem.filename}. Reason: {e}")
                traceback.print_exc()
            
            print("-" * 80)

def main():
    args = load_config("configs/tritonbench_oneshot_config.yaml")
    model = KimiK2Model(api_key=args.api_key, model_id=args.model_id)
    
    result_path = None
    dataset = TritonBench(statis_path=args.statis_path, 
                          py_folder=args.py_folder, 
                          instruction_path=args.instruction_path,
                          py_interpreter=args.py_interpreter, 
                          golden_metrics=args.golden_metrics,
                          perf_ref_folder=args.perf_ref_folder,
                          perf_G_path=args.perf_G_path,
                          result_path=result_path,
                          target_kernels=args.target_kernels)

    # <<<<<<<< INJECTING AGENT 2 TEST LOGIC >>>>>>>>
    run_agent_2_test(model, dataset)
    print("\nTest finished. Exiting.")
    exit()
    # <<<<<<<<<<<<<<<<<<<->>>>>>>>>>>>>>>>>>

    agent = Reflexion_Oneshot(model=model, dataset=dataset, corpus_path=args.corpus_path)
    agent.run(output_path=args.output_path, multi_thread=args.multi_thread, iteration_num=args.max_iteration, temperature=args.temperature, datalen=None)

if __name__ == "__main__":
    main()
