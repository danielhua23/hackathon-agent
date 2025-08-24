
import os
# from agents.reflexion_oneshot import Reflexion_Oneshot
from agents.multi_agent_pipeline import MultiAgentPipeline # Use our new agent
from models.KimiK2 import KimiK2Model
from dataloaders.TritonBench import TritonBench
from args_config import load_config


def main():
    # After fixing the paths inside TritonBench, this main function can be clean again.
    args = load_config("configs/tritonbench_oneshot_config.yaml")

    # --- Temporarily override for a quick 2-iteration test run ---
    args.max_iteration = 2
    print(f"--- INFO: Running a quick test with max_iteration = {args.max_iteration} ---")

    # setup LLM model
    model = KimiK2Model(api_key=args.api_key, model_id=args.model_id)
    
    # setup dataset
    # Now that TritonBench handles path correction internally, we can pass the raw config paths.
    result_path = None
    
    # --- Targeted Test Run for Failed Kernels ---
    # We will only run the two kernels that failed in the last full test.
    targeted_kernels = None # Set to None to run all kernels
    print(f"--- INFO: Running a full test for all 10 kernels. ---")

    dataset = TritonBench(statis_path=args.statis_path, 
                          py_folder=args.py_folder, 
                          instruction_path=args.instruction_path,
                          py_interpreter=args.py_interpreter, 
                          golden_metrics=args.golden_metrics,
                          perf_ref_folder=args.perf_ref_folder,
                          perf_G_path=args.perf_G_path,
                          result_path=result_path,
                          target_kernels=targeted_kernels) # Pass the targeted list here

    # setup agent
    agent = MultiAgentPipeline(model=model, dataset=dataset, corpus_path=args.corpus_path)

    # run the agent for a single iteration
    agent.run(output_path=args.output_path, 
            multi_thread=False, 
            iteration_num=1, # We only need to check the first iteration
            temperature=args.temperature, 
            datalen=None) # Datalen should be None to run all items in the (now filtered) dataset


if __name__ == "__main__":
    main()
