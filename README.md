## Technical report [![arXiv](https://img.shields.io/badge/arXiv-2507.23194-b31b1b.svg)](https://arxiv.org/abs/2507.23194)

## Introduction

This is an LLM-based multi-agent framework, which can generate functional and efficient gpu kernels automatically.

The framework is extendable and flexible. You can easily make you own coding agent and test it on our GEAK-eval https://github.com/AMD-AIG-AIMA/GEAK-eval.
We provide a baseline agent to let you run directly.

## GEAK-agent
<img width="608" height="327" alt="image" src="https://github.com/user-attachments/assets/a16f022e-6371-45ac-8159-59abf9df4972" />

It contains a Generator, a Reflector, an Evaluator and an Optimizer. The actor generates codes according to the query and context information. The Reflector is responsible for reflecting on the generated code and the error trace if the code failed to run. The Evaluator has a cascade structure. It tests the generated code for the functionality first. If the generated code doesn't pass the functionality test, the error trace will be fedback to the Reflector. Otherwise, the Evaluator will evaluate the performance including latency and efficiency. The Optimizer gets the generated codes, which pass the evaluator's tests, and gives a strategy to optimize the code in terms of latency and efficiency.

### Core Process Improvements: Building a Robust Iterative Optimization System

Our framework introduces a structured, multi-agent approach that separates concerns into a four-stage pipeline for the initial code generation, followed by a robust, two-step process for iterative reflection and repair.

#### 1. The Four-Agent Pipeline for Initial Generation

Instead of relying on a single agent, we decompose the complex task of kernel generation into a "chain of thought" executed by four specialized agents:
1.  **Agent 1: The Analyst**: Receives the problem description and performs a deep analysis of the requirements, constraints, and potential challenges.
2.  **Agent 2: The Baseline Implementer**: Takes the Analyst's report and generates a simple, functionally correct baseline implementation of the kernel. This serves as a solid foundation for optimization.
3.  **Agent 3: The Strategist**: Analyzes the baseline code and proposes a list of concrete, high-level optimization strategies (e.g., "increase block size," "apply autotuning").
4.  **Agent 4: The Executor**: Takes one strategy at a time and applies it to the baseline code, generating the final, optimized kernel for the first iteration.

This structured pipeline ensures a high-quality initial code generation, setting the stage for effective iterative refinement.

#### 2. The "Code Goalkeeper": Automated Signature Correction

**Problem:** A primary reason for evaluation failures was the LLM's tendency to generate code with incorrect function signatures (e.g., wrong function name, incorrect parameter order). Constraining the model via prompts proved to be unreliable.

**Solution:** We introduced a "Code Goalkeeper," a deterministic post-processing step that runs immediately after code generation. This mechanism uses Python's Abstract Syntax Tree (`ast`) module to:
1.  Parse the generated code and the baseline code provided by an earlier agent.
2.  Perform a **safety check**: It verifies that the parameters of the generated function are a superset of the baseline function's parameters. This prevents catastrophic failures if the core logic has been fundamentally altered.
3.  If the check passes, it programmatically replaces the signature of the generated function with the "golden" signature from the baseline code.

**Impact:** This completely decouples the LLM's creative optimization task from the rigid, mechanical task of format compliance. It dramatically increases the `Call Status: True` rate, allowing the iterative process to focus on deeper logic and performance issues.

#### 3. The "Expert Diagnostician": A Two-Step Reflection and Repair Process

**Problem:** The original reflection process was inefficient. It would feed a raw error log back to the model, which often struggled to identify the root cause, leading to repeated, ineffective repair attempts.

**Solution:** We re-architected the reflection phase into a structured, two-step "Diagnose-and-Repair" workflow:
1.  **Step 1: Expert Diagnosis:** When a test fails, we no longer use a generic reflection prompt. Instead, we use a specialized **"Expert Diagnostician" prompt**. This prompt guides the LLM to act like a senior GPU kernel engineer. It analyzes the specific error type (`Runtime Error`, `Correctness Error`, `Poor Performance`) and the traceback to produce a concise, high-level **Correction Plan**.
2.  **Step 2: Guided Repair:** This newly generated Correction Plan is then passed to a dedicated **"Code Repair" prompt**. Instead of grappling with a raw error log, this agent receives clear, expert-level instructions, enabling it to perform a much more precise and effective code fix.

**Impact:** This structured process transforms the reflection loop from a vague, trial-and-error cycle into a focused, expert-driven debugging session. It significantly improves the agent's ability to recover from "second-layer" failures (logic and performance bugs) after the Goalkeeper has handled the initial formatting issues.

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
