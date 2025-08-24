import os
from tqdm import tqdm
from loguru import logger
import json
from dataclasses import asdict
from agents.Reflexion import Reflexion
from utils.utils import extract_function_signatures, clear_code, extract_function_calls
from prompts import prompt_for_reflection
from memories.Memory import MemoryClassMeta
from models.Base import BaseModel
from retrievers.retriever import BM25Retriever
from prompts import prompt_for_generation
from concurrent.futures import ThreadPoolExecutor, as_completed



class Reflexion_Oneshot(Reflexion):
    """
    1. Given the instructions and contexts, the LLM generates candidate solutions.
    2. The generated solutions are executed to produce results or, in case of failure, error messages.
    3. If execution fails, the LLM analyzes the error trace and suggests a fix.
    4. The process repeats, with the LLM generating improved solutions based on previous feedback.
    """

    def __init__(self, model: BaseModel, dataset, corpus_path, mem_file=None):
        self.model = model
        self.dataset = dataset
        self.memories = []

        self.instruction_retriever = BM25Retriever()
        self.instruction_retriever.process(content_input_path=corpus_path)
        self.code_retriever = BM25Retriever(mode="code")
        self.code_retriever.process(content_input_path=corpus_path)

        self.memory_init(mem_file)

    def memory_init(self, mem_file=None):
        class Memory(metaclass=MemoryClassMeta, field_names=["ps", 
                                                             "err_msg", 
                                                             "reflection", 
                                                             "function_signatures", 
                                                             "oneshot",
                                                             "pass_call", 
                                                            ]):
            pass
        
        if mem_file is not None:
            assert mem_file.endswith(".json"), f"expect a json file, but got {mem_file} instead"
            with open(mem_file, "r") as f:
                input_mems = json.load(f)
            assert len(input_mems) == len(self.dataset), f"expect {len(self.dataset)} samples, but got {len(input_mems)} instead"

        for ps in self.dataset.problem_states:
            if ps.label:
                fs_mem = extract_function_signatures(ps.label)
            else:
                fs_mem = None
            if mem_file is None:
                os_mem = self.instruction_retriever.query(ps.instruction)[0]
                tmp_mem = Memory(ps=ps, 
                                err_msg=None, 
                                reflection=None, 
                                function_signatures=fs_mem, 
                                oneshot=os_mem["code"],
                                pass_call=False,
                                )
            else:
                input_mem = input_mems[ps.filename]
                tmp_mem = Memory(ps=ps, 
                                err_msg=input_mem["err_msg"], 
                                reflection=input_mem["reflection"], 
                                function_signatures=fs_mem, 
                                oneshot=input_mem["oneshot"],
                                pass_call=input_mem["pass_call"],
                                )
            self.memories.append(tmp_mem)

    def run(self, output_path=None, multi_thread=True, verbose=False, datalen=None, iteration_num=0, temperature=0):
        data_len = datalen if datalen else len(self.dataset)
        for iter in range(iteration_num):
            # Filter only failed kernels for this iteration (correctness check)
            failed_memories = [mem for mem in self.memories[:data_len] if not mem.pass_call]
            
            if not failed_memories:
                logger.info(f"\n=== All kernels passed, stopping at iteration {iter} ===")
                break
                
            logger.info(f"\n=== Iteration {iter} ===")
            logger.info(f"Processing {len(failed_memories)} failed kernels out of {data_len} total")
            
            if output_path is not None:
                root, extension = os.path.splitext(output_path)
                iter_path = f"{root}_{iter}{extension}"

            if multi_thread:
                thread_num = 3
            
            # generate solution for failed kernels only
            logger.info(f"\ngenerate solution for failed kernels")
            with tqdm(total=len(failed_memories)) as pbar:
                if multi_thread:
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_solution, mem, temperature): mem for mem in failed_memories}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in failed_memories:
                        self.generate_solution(mem, temperature=temperature)
                        pbar.update(1)
            
            logger.info(f"\nrun correctness tests on gpu")
            for mem in tqdm(failed_memories):
                try:
                    pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr = self.dataset.test_opt_correctness(
                        mem.ps.solution, mem.ps.filename, "temp", exe_dir="pass_exe"
                    )
                except Exception as e:
                    logger.info(f"failed to test the code due to : {e}")
                    mem.err_msg = f"failed to test the code due to: {e}"
                    continue
                
                if not pass_call:
                    mem.err_msg = call_stderr
                elif not pass_exe:
                    mem.err_msg = exe_stderr
                else:
                    # Both call and execution passed - mark as successful
                    mem.pass_call = True
                    mem.err_msg = None  # Clear previous error
            """
            To measure kernel latency, follow these steps:

            self.dataset.write_perf_file(input_folder_path=exe_dir, results_path=perf_result_dir, tmp_dir=script_dir)
            self.dataset.run_perf_scripts(gpu_id=gpu_id, script_dir=script_dir, log_dir=perf_log_dir)

            for mem in self.memories[:data_len]:
                path_gen = os.path.join(perf_result_dir, mem.ps.filename[:-3] + ".json")
                if not os.path.exists(path_gen):
                    continue
                try:
                    _, efficiency, ms = self.dataset.calculate(path_gen, path_ref=None)

                    print(f"{mem.ps.filename} latency: {ms}")
                    print(f"{mem.ps.filename} efficiency: {efficiency}\n")
                    
                    
                except Exception as e:
                    print(f"{mem.ps.filename} failed due to {e}")

            """

            # generate reflections for failed kernels only
            logger.info(f"\ngenerate reflections for failed kernels")
            still_failed_memories = [mem for mem in failed_memories if not mem.pass_call]
            with tqdm(total=len(still_failed_memories)) as pbar:
                if multi_thread:
                    with ThreadPoolExecutor(max_workers=thread_num) as executor:
                        futures = {executor.submit(self.generate_reflexion, mem, temperature): mem for mem in still_failed_memories}
                        for future in as_completed(futures):
                            pbar.update(1)
                else:
                    for mem in still_failed_memories:
                        self.generate_reflexion(mem, temperature=temperature)
                        pbar.update(1)
            
            if output_path is not None:
                self.dataset.write_file(iter_path)
                    

    
    def generate_solution(self, mem, temperature=0):
        if mem.pass_call:
            logger.debug(f"Skipping {mem.ps.filename} - already passed")
            return
        
        # tab = "\n"
        # fss_text = "".join(f"* {sig}{tab}" for sig in mem.function_signatures)
        text = prompt_for_generation.prompt.format(
            instruction=mem.ps.instruction,
            function_signatures=""
        )

        if not mem.ps.solution:
            text += f"\nHere is an example snippet of code: {mem.oneshot}"
        else:
            one_shot = self.code_retriever.query(mem.ps.solution)[0]["code"]
            text += f"\nHere is an example snippet of code: {one_shot}"
            text += f"\nPrevious attempt implementation:{mem.ps.solution}"
            
                  
        if mem.err_msg:
            text += f"\nTest messages for previous attempt:{mem.err_msg}"
        
        if mem.reflection:
            text += f"\nReflection on previous attempt:{mem.reflection}"

        text += "Please output the codes only without explanation, which we can run directly."
        msg = [
            {"role": "user", "content": text},
        ]
        response = self.model.generate(msg, temperature=temperature)
        mem.ps.solution = clear_code(response)

        return



    def generate_reflexion(self, mem, temperature):
        if mem.pass_call:
            logger.debug(f"Skipping reflection for {mem.ps.filename} - already passed")
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