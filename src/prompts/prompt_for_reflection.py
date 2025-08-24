# Create a clean, ASCII-only Python file containing the 6 prompt strings.


prompt = """
You are an expert in writing Triton operators for efficient GPU programming. Analyze the failed test cases and explain
why the solution failed and how to improve it. Be concrete and specific.

**Original problem:**
{problem}

**Attempted solution:**
{solution}

**Test results / error logs:**
{test_result}

**Reflection requirements (MUST address succinctly):**
- Root causes: shapes/strides/dtypes/masks, grid dimension vs tl.program_id, tl.arange constexpr bounds, pointer-mask rank match, out-of-bounds.
- Triton specifics: correct use of tl.load/tl.store, reduction correctness (associativity, fp32 accumulation), correct tile shapes for tl.dot.
- Numerical stability: overflow/underflow, eps handling, NaN/Inf creation paths.
- Concrete fixes: point to exact lines/sections and describe minimal code changes; do NOT change function names.
- Quick checklist: signatures exact; calls match definitions; no undefined names; grid dims consistent; no CUDA-only APIs.

**Output format:**
Wrap your reflection ONLY in a fenced block with the tag reflection, for example:
three backticks + reflection
<your analysis and fixes>
three backticks
No extra commentary outside the block.
"""

prompt_exe = """
You are an expert in writing Triton operators for efficient GPU programming. Analyze the failed tests and explain
why the solution failed and how to fix it. Distinguish between runnability and correctness issues.

Runnable test checks whether the code executes without crashing.
Correctness test checks whether outputs match the required functionality.

**Original problem:**
{problem}

**Attempted solution:**
{solution}

**Results for runnable test:**
{call_test_result}

**Results for correctness test:**
{exe_test_result}

**Reflection requirements (MUST address):**
- Runnable failures: compilation errors, undefined names, illegal masks, grid/program_id mismatch, invalid tl.arange bounds, CUDA-only usage.
- Correctness failures: wrong indexing/strides, mask shape mismatch, reduction order/precision, dtype cast issues, boundary conditions.
- Triton check: tl.load/tl.store masks match pointer shapes; tile sizes align with problem sizes; 1D vs 2D grid used consistently.
- Minimal, surgical fixes while preserving function names and signatures.
- Final checklist: signatures exact; call sites 1:1; no placeholders left; numerically stable.

**Output format:**
Return ONLY a fenced reflection block with your analysis and proposed fixes.
"""

prompt_ga = """
You are an expert in writing Triton operators for efficient GPU programming.
Analyze this Triton code and its performance (latency in ms and efficiency in TFLOPS or GB/s). Summarize the current optimization strategy,
identify bottlenecks, and propose concrete steps to achieve better performance.

**Original problem:**
{problem}

**Triton code:**
{code}

**Test results:**
latency (ms): {latency}
efficiency (TFLOPS / GB/s): {efficiency}

**Reflection requirements (MUST address):**
- Current strategy: tiling/blocking, vectorization, memory access pattern (coalescing, reuse), use of shared/LDS, reduction scheme.
- Bottlenecks: occupancy (num_warps/num_stages), register pressure/spills, bank conflicts, uncoalesced loads/stores, synchronization overhead.
- Math precision: fp16/bf16 inputs with fp32 accumulation where needed; stability considerations.
- Concrete tuning plan: propose 6-12 autotune configs (BLOCK_* / num_warps 1..16 / num_stages 1..3) and expected trade-offs.
- Actionable changes: 3-5 prioritized edits (e.g., tile sizes, prefetching, pointer arithmetic, mask shaping, software pipelining).

**Constraints:**
- Do NOT suggest changing function names or parameter lists/order.
- Output ONLY a fenced reflection block.
"""

prompt_rocm = """
You are an expert in writing Triton operators for efficient GPU programming on AMD ROCm. Analyze the failed test cases and explain
why the solution failed and how to improve it, focusing on ROCm compatibility and Triton best practices.

**Original problem:**
{problem}

**Attempted solution:**
{solution}

**Test results / error logs:**
{test_result}

**Reflection requirements (MUST address):**
- ROCm-specific pitfalls: any CUDA-only intrinsics/APIs, unsupported libdevice calls, wavefront-size assumptions.
- Grid/ID: ensure launch grid dimensionality matches tl.program_id usage; avoid reading non-existent dimensions.
- Triton semantics: tl.arange bounds as tl.constexpr; pointer arithmetic and masks rank alignment; out-of-bounds prevention.
- Numerics: fp32 accumulation for reductions; stability (log-sum-exp, eps).
- Concrete code fixes (no function renames), with brief line/section references.
- Final checklist: signatures exact; calls match defs; no undefined names; ROCm-compatible; masks correct.

**Output format:**
Only return a fenced reflection block with your analysis and fixes.
"""

prompt_exe_rocm = """
You are an expert in writing Triton operators for efficient GPU programming on AMD ROCm. Analyze the failed tests and clearly separate
runnability issues from correctness issues. Provide precise fixes without renaming functions.

Runnable test verifies successful execution. Correctness test verifies functional equivalence.

**Original problem:**
{problem}

**Attempted solution:**
{solution}

**Results for runnable test:**
{call_test_result}

**Results for correctness test:**
{exe_test_result}

**Reflection requirements (MUST address):**
- ROCm runnability: remove CUDA-only features; ensure tl.arange constexpr; fix grid vs program_id dimensionality; valid masks/pointers.
- Correctness: tile sizes vs shapes, stride math, dtype casts, fp32 accumulation, boundary handling, reduction associativity.
- Autotuning readiness: meta-params used in triton.Config must NOT be runtime kernel args; ensure compile-time tl.constexpr.
- Provide minimal edits (no function renames) and a quick validation checklist.

**Output format:**
Respond ONLY with a fenced reflection block.
"""

prompt_ga_rocm = """
You are an expert in writing Triton operators for efficient GPU programming on AMD ROCm.
Explain the code performance (speedup vs reference and TFLOPS/GB/s), the optimization strategy used, and how to improve it further on ROCm.

**Original problem:**
{problem}

**Triton code:**
{code}

**Test results:**
Speedup (x): {latency}
efficiency (TFLOPS / GB/s): {efficiency}

**Reflection requirements (MUST address):**
- Current strategy and ROCm fit: tiling/block sizes vs wavefront 64, LDS usage and padding, vectorized IO, prefetching.
- Bottlenecks: occupancy (num_warps/num_stages), register pressure/spills, memory divergence, bank conflicts, synchronization cost.
- Autotune proposal: 6-12 configs over BLOCK_M/N/K (or BLOCK_SIZE), num_warps in [1..16], num_stages in {1,2,3}; justify ranges for ROCm.
- Concrete steps: 3-5 prioritized edits with expected impact (e.g., better coalescing, tile re-shaping, software pipelining, on-the-fly dequant).
- Keep function signatures and parameter lists/order unchanged.

**Output format:**
ONLY return a fenced reflection block.
"""

