## Technical report [![arXiv](https://img.shields.io/badge/arXiv-2507.23194-b31b1b.svg)](https://arxiv.org/abs/2507.23194)

## Introduction

This is an LLM-based multi-agent framework, which can generate functional and efficient gpu kernels automatically.

The framework is extendable and flexible. You can easily make you own coding agent and test it on our GEAK-eval https://github.com/AMD-AIG-AIMA/GEAK-eval.
We provide a baseline agent to let you run directly.

## GEAK-agent
<img width="608" height="327" alt="image" src="https://github.com/user-attachments/assets/a16f022e-6371-45ac-8159-59abf9df4972" />

It contains a Generator, a Reflector, an Evaluator and an Optimizer. The actor generates codes according to the query and context information. The Reflector is responsible for reflecting on the generated code and the error trace if the code failed to run. The Evaluator has a cascade structure. It tests the generated code for the functionality first. If the generated code doesn't pass the functionality test, the error trace will be fedback to the Reflector. Otherwise, the Evaluator will evaluate the performance including latency and efficiency. The Optimizer gets the generated codes, which pass the evaluator's tests, and gives a strategy to optimize the code in terms of latency and efficiency.

### the Optimizer
We provide previous generated codes as reference codes with their corresponding performance to the Optimizer. The number of reference codes is controlled by the arg  `ancestor_num`. The reference codes are arranged in ascending order to help the Optimizer LLM find the optimization direction. We don't ask the LLM to generate new codes directly from the reference codes, instead we ask the Optimizer to analyze the reference codes first and to generate a promising strategy to optimize the code. Then we feed the generated optimization stratgey to the Generator to generate new codes.

### debugging trap
LLMs frequently get caught in debugging traps. When their generated code has bugs, we provide the error trace to the Reflector correction. However, we've observed that sometimes code can undergo several reflection cycles while still being plagued by the same bug. We refer to this as a debugging trap.

To prevent the LLM from getting stuck in a debugging trap, we limit debugging attempts per code snippet using `max_perf_debug_num`. If the code fails after this many fixes, the agent must abandon the current approach and generate a fresh strategy and code.

## Run the Agents

pls follow the guidelines in https://github.com/danielhua23/ai_sprint_shanghai/tree/main/hackathon_guides/track2-agent

#### Citation
If you find this work useful in your research or applications, please consider citing:

```bibtex
@misc{wang2025geakintroducingtritonkernel,
      title={Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks}, 
      author={Jianghui Wang and Vinay Joshi and Saptarshi Majumder and Xu Chao and Bin Ding and Ziqiong Liu and Pratik Prabhanjan Brahma and Dong Li and Zicheng Liu and Emad Barsoum},
      year={2025},
      eprint={2507.23194},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.23194}, 
}
```
