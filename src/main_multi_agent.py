import os
# --- Switch Agent Class ---
# from agents.reflexion_oneshot import Reflexion_Oneshot
from agents.multi_agent_pipeline import MultiAgentPipeline
from models.KimiK2 import KimiK2Model
from dataloaders.TritonBench import TritonBench
from args_config import load_config


def main():
    args = load_config("configs/tritonbench_oneshot_config.yaml")

    # For a quick test, let's limit the number of iterations and kernels
    args.max_iteration = 3
    test_cases_to_run = 2
    print(f"--- RUNNING WITH A TEST CONFIG: max_iteration = {args.max_iteration}, kernels = {test_cases_to_run} ---")

    # setup LLM model
    model = KimiK2Model(api_key=args.api_key, model_id=args.model_id)
    
    # setup dataset
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

    # setup agent - switch to our new pipeline agent
    # We now call our agent with the original arguments it expects, including corpus_path.
    agent = MultiAgentPipeline(model=model, dataset=dataset, corpus_path=args.corpus_path)

    # run the agent
    agent.run(output_path=args.output_path, 
            multi_thread=True,
            iteration_num=args.max_iteration, 
            temperature=args.temperature,
            datalen=test_cases_to_run)


if __name__ == "__main__":
    main()
