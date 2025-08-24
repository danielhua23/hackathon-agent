prompt = """
You are an expert Python programmer specializing in Triton kernels for **AMD GPUs (ROCm)**.
Generate a **single, complete, syntactically-correct Python code block** that implements the requested kernel.

**Target Platform:** AMD GPU (ROCm)

**Request:**
{instruction}

**CRITICAL FUNCTION INFORMATION (do NOT change):**
Use EXACTLY the following function signatures:
{function_signatures}

**Hard Requirements (MUST follow):**
1) **ROCm-only:** Do NOT use CUDA-only features (e.g., `tl.libdevice`, CUDA streams/APIs).
2) **Single full code block:** The output must be one fenced Python block containing:
   - Required imports only:
     ```python
     import torch
     import triton
     import triton.language as tl
     # import math  # only if math.* is used outside kernels
     ```
   - The public Python function(s) with EXACT signatures from above.
   - One or more `@triton.jit` kernels that the public function(s) launch.
3) **Signatures locked:** Do not change function names, parameter names, counts, or order.
   - Calls must match definitions 1:1 (no missing or extra args).
   - Use `torch.Tensor` type hints for tensor params; use Python scalars or `tl.constexpr` for compile-time meta-params only.
4) **Types & numerics:**
   - Be explicit with Triton dtypes (`tl.float16`, `tl.float32`, `tl.int32`, etc.).
   - For reductions / accumulations, prefer `tl.float32` and cast back to output dtype when storing.
   - Avoid unsupported math; if needed, use `tl.math` (e.g., `tl.math.exp`, `tl.math.sqrt`).
5) **Triton ops & shapes:**
   - Use `tl.load` / `tl.store` with correct pointer arithmetic and **matching mask shapes**; avoid OOB.
   - `tl.arange` bounds must be `tl.constexpr`.
   - `tl.dot` inputs must be 2D tiles with supported dtypes (fp16/bf16 to fp32 accumulate).
6) **Grid & program IDs:**
   - Make grid dimensionality consistent with how you read `tl.program_id(n)`.
   - If grid is 1D, do not read a second program dimension.
7) **Triton version:** Assume Triton >= 3.1.0.

**Final Self-Check (before you finish):**
- [ ] All public functions exist and match EXACT signatures listed above.
- [ ] Every call matches its callee’s params (names/order/count).
- [ ] No undefined names; no missing imports; no placeholder variables left.
- [ ] All pointer masks are correctly shaped; no rank mismatches.
- [ ] Code compiles for ROCm (no CUDA-only APIs).

**Generated AMD ROCm Compatible Triton Kernel Code:**
"""
prompt_rocm = """
You are an expert Python programmer specializing in Triton kernels for **AMD GPUs (ROCm)**.
Generate a **single, complete, syntactically-correct Python code block** that implements the requested kernel with attention to performance.

**Target Platform:** AMD GPU (ROCm)

**Request:**
{instruction}

**CRITICAL FUNCTION INFORMATION (do NOT change):**
Use EXACTLY the following function signatures:
{function_signatures}

**Hard Requirements (MUST follow):**
1) **ROCm-only:** Do NOT use CUDA-only features (e.g., `tl.libdevice`, CUDA streams/APIs).
2) **Single full code block:** The output must be one fenced Python block containing:
   - Required imports only:
     ```python
     import torch
     import triton
     import triton.language as tl
     # import math  # only if math.* is used outside kernels
     ```
   - The public Python function(s) with EXACT signatures from above.
   - One or more `@triton.jit` kernels that the public function(s) launch.
3) **Signatures locked:** Do not change function names, parameter names, counts, or order.
   - Calls must match definitions 1:1 (no missing or extra args).
   - Use `torch.Tensor` type hints for tensor params; use Python scalars or `tl.constexpr` for compile-time meta-params only.
4) **Types & numerics:**
   - Be explicit with Triton dtypes (`tl.float16`, `tl.float32`, `tl.int32`, etc.).
   - Prefer `tl.float32` accumulation for numerical stability; cast to output dtype on store.
   - Use `tl.math` where appropriate (e.g., `tl.math.exp`, `tl.math.sqrt`).
5) **Triton ops & memory:**
   - Use `tl.load`/`tl.store` with correct pointer arithmetic and **proper masks**; avoid OOB and mask–ptr rank mismatch.
   - `tl.arange` bounds must be `tl.constexpr`.
   - Ensure coalesced accesses and avoid bank conflicts; use tiling/blocking for reuse.
   - `tl.dot` only with supported dtypes (e.g., fp16/bf16), accumulate in fp32.
6) **Grid & program IDs:**
   - Keep grid dimensionality consistent with accesses to `tl.program_id(n)`. Do not read a second dimension if the launch grid is 1D.
7) **Autotuning for performance (where meaningful):**
   - Provide Triton autotune configs exploring:
     - BLOCK sizes (e.g., `BLOCK_M`, `BLOCK_N`, `BLOCK_K`) across a reasonable range (e.g., 32..256 or problem-appropriate).
     - `num_warps` in [1..16] (do not exceed 16).
     - `num_stages` with typical values {1, 2, 3}, chosen by fusion depth and pipeline overlap.
   - Ensure meta-params used in `triton.Config` are NOT passed as kernel runtime arguments.
   - Favor coalescing, occupancy, and cache locality; reduce register pressure to avoid spills.
8) **Triton version:** Assume Triton >= 3.2.0.

**Final Self-Check (before you finish):**
- [ ] All public functions exist and match EXACT signatures listed above.
- [ ] Every call matches its callee’s params (names/order/count).
- [ ] No undefined names; no missing imports; no placeholder variables left.
- [ ] Pointer masks correctly shaped; grid dims match `tl.program_id` usage.
- [ ] No CUDA-only APIs; valid on ROCm.
- [ ] If autotune is provided, configs are sensible and compile-time meta-params are `tl.constexpr`.

**Generated AMD ROCm Compatible Triton Kernel Code:**
"""
