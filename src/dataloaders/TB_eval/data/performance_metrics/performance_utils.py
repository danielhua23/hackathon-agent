import torch
import triton
import triton.language as tl

from typing import Callable
import json
import os

class do_bench_config():
    def __init__(
            self,
            warm_up=25,
            repetition=100,
            grad_to_none=None,
            quantiles=[0.5, 0.8, 0.2],
            return_mode="median"
    ):
        self.warm_up = warm_up
        self.repetition = repetition
        self.grad_to_none = grad_to_none
        self.quantiles = quantiles
        self.return_mode = return_mode

class Performance_Metrics:
    def __init__(
            self,
            op_name,
            dtype=None,
            is_backward=False,
            **kwargs
    ):
        self.op_name = op_name
        self.dtype = dtype
        if is_backward:
            self.op_name += 'backward'
        self.kwargs = kwargs

        self.input_tensors = []
        self.do_bench_config = do_bench_config()
        # 添加folder_path属性，默认值会在run_benchmark中被替换
        self.folder_path = "../outputs/optimagent_gpt41_tritonbench_perf_results"

    def get_input_tensors(self):
        raise NotImplementedError("You must implement this method to get input tensors")

    def to_cuda(self, input_tensor):
        raise NotImplementedError("You must implement this method to move tensors to CUDA")
    
    def call_op(self, input_tensor):
        raise NotImplementedError("You must implement this method to call the op")

    def get_do_bench_config(self, warmup=None, rep=None):
        if warmup != None and rep != None:
            self.do_bench_config = do_bench_config(
                warm_up=warmup,
                repetition=rep,
            )
            return

        if self.input_tensors == []:
            raise NotImplementedError("You must implement this method to get input_tensors")
        
        previous_ms = None
        epsilon = 1e-4
        stable_count = 0
        max_stable_count = 3
        input_tensor = self.to_cuda(self.input_tensors[-1])

        for t in range(1, 11):
            warmup = 100 * t
            rep = 1000 * t
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: self.call_op(input_tensor),
                warmup=warmup,
                rep=rep,
                quantiles=[0.5, 0.8, 0.2],
                return_mode="median"
            )

            print("warmup time:", warmup, "rep time:", rep, "runtime:", ms)

            if previous_ms is not None:
                relative_change = abs(ms - previous_ms) / abs(previous_ms) if previous_ms != 0 else float('inf')

                if relative_change < epsilon:
                    stable_count += 1
                else:
                    stable_count = 0
            
            if stable_count >= max_stable_count:
                print(f"MS stabilized with warmup={warmup} and rep={rep}")
                self.do_bench_config = do_bench_config(
                    warm_up=warmup,
                    repetition=rep,
                )
                return

            previous_ms = ms
        
        print("MS did not stabilize. Returning default config.")
        # Instead of raising an error, set default config
        self.do_bench_config = do_bench_config()
        return

    def get_runtime(self, op: Callable):
        ms, min_ms, max_ms = triton.testing.do_bench(
            op,
            warmup=self.do_bench_config.warm_up,
            rep=self.do_bench_config.repetition,
            quantiles=self.do_bench_config.quantiles,
            return_mode=self.do_bench_config.return_mode
        )
        return ms
    
    def get_gbps(self, input_tensor, runtime):
        raise NotImplementedError("You must implement this method to get the method to calculate GBPS")

    def get_tflops(self, input_tensor, runtime):
        raise NotImplementedError("You must implement this method to get the method to calculate TFLOPS")

    def run_benchmark(self):
        results = []
        for input_tensor_ in self.input_tensors:
            try:
                input_tensor = self.to_cuda(input_tensor_)
                # print(input_tensor)
                input_tensor_clone = input_tensor  # Create a copy to avoid late binding issues
                op = lambda : self.call_op(input_tensor_clone)            
                ms = self.get_runtime(op)
                gbps = self.get_gbps(input_tensor, ms)
                tflops = self.get_tflops(input_tensor, ms)
                result = {
                    "input_size": [item.shape if type(item)==torch.Tensor else item for item in input_tensor],
                    "ms": ms,
                    "GB/s": gbps,
                    "TFLOPS": tflops
                }
                print(result)
                results.append(result)
            except Exception as e:
                print(f"Failed to run benchmark for input tensor. Error: {e}")
            input_tensor = None
            
        # 确保folder_path目录存在
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path, exist_ok=True)
            
        file_name = self.op_name + ".json"
        file_path = os.path.join(self.folder_path, file_name)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=4)