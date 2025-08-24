
import os
from agents.reflexion_oneshot import Reflexion_Oneshot
from models.KimiK2 import KimiK2Model
from dataloaders.TritonBench import TritonBench
from args_config import load_config
from loguru import logger
import sys, time
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(f"logs/reflexion_oneshot_debug_{time.strftime('%Y%m%d_%H%M%S')}.log", level="DEBUG")

def main():
    args = load_config("configs/tritonbench_oneshot_config.yaml")

    # setup LLM model
    #model = OpenAIModel(api_key=args.api_key, model_id=args.model_id)
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

    # setup agent
    agent = Reflexion_Oneshot(model=model, dataset=dataset, corpus_path=args.corpus_path)

    # run the agent
    agent.run(output_path=args.output_path, multi_thread=args.multi_thread, iteration_num=args.max_iteration, temperature=args.temperature, datalen=None)


if __name__ == "__main__":
    main()
