# improved_prompt_for_reflection_v5.py
# Enhanced with Human Priors, Golden-aware fix order, and 10-task personalized debugging cards.

prompt = r"""
You are an expert in Triton operators on **AMD ROCm**. Analyze failed tests and explain precisely **why** it failed and **how** to fix it. Be specific, actionable, and do **NOT** propose renaming functions.

Original problem:
{problem}

Attempted solution:
{solution}

Test results:
{test_result}

DIAGNOSIS CHECKLIST (ROCm + Wave64)
1) Kernel shape & grid
   - tl.program_id rank matches launch grid rank (1D vs 2D).
   - tl.arange bounds are tl.constexpr; no dynamic ranges.
   - If a loop needs runtime iters, replace with `tl.static_range(UPPER)` + per-iter mask.

2) Pointers & masks
   - Correct row/col stride usage for **each** tensor (X/Y/DX, A/B/C, etc.).
   - Mask shapes match tiles; no unintended broadcasting.
   - Index math in hot loops uses tl.int32.

3) Dtypes & numerics
   - tl.dot inputs are fp16/bf16 2D tiles; fp32 accumulation enabled.
   - eps in divides/norms; online max-sub for softmax-like ops.
   - **Elementwise sin** uses tl.sin (not tl.math.sin).
   - **Bit ops** on integers; cast packed int4 to int32 prior to shifts; ensure BLOCK_K%8==0 for INT4.

4) Autotune discipline
   - SAFE→AGGRESSIVE ladder with multiple triton.Config candidates; key includes shapes/dtypes/flags.
   - BLOCK_* must be tl.constexpr kernel params (never tensors from wrapper).

5) Wave64 performance hygiene
   - num_warps ∈ {2,4,8,16}; tiles multiples of 16; GROUP_M for L2 reuse.
   - Watch for register spills (symptom: perf collapse vs smaller tiles).

SCORE-AWARE FIX ORDER
- (1) Minimal edits to **pass correctness** (mask rank, strides, dtype casts, constexpr bounds, replace dynamic loops).
- (2) Then propose config ladder upgrades that usually beat Golden.

Return only a code block with tag reflection summarizing:
1) Concrete root causes (quote code lines/symptoms).
2) Minimal, targeted fixes (grid tuple, tl.constexpr, masks, strides, casts, eps).
3) Immediate MI-series tuning (tiles/warps/stages + autotune key).
"""

prompt_exe = r"""
You are an expert in Triton on **AMD ROCm**. Two tests were run: runnable (compile/launch) and correctness (numerics). Explain exactly why it failed and how to fix it **without** renaming functions.

Original problem:
{problem}

Attempted solution:
{solution}

Results for runnable test:
{call_test_result}

Results for correctness test:
{exe_test_result}

Return only a code block with tag reflection:

- Runnable failures
  - Check: grid rank vs tl.program_id usage; tl.arange constexpr; wrong mask rank; pointer/stride math; passing BLOCK_* from wrapper; CUDA-only calls; missing out allocation; dynamic Python loop in kernel.
  - Fix: grid = (triton.cdiv(M,BM), triton.cdiv(N,BN)); cast loop indices to tl.int32; mask shape [BM,BN]; correct pointer formula; remove CUDA-only bits; make BLOCK_* tl.constexpr; allocate `out` when None; replace dynamic `range(...)` with `tl.static_range(UPPER)` + mask.

- Correctness failures
  - Check: wrong reduction axis; tl.dot type/shape; missing fp32 acc; tail off-by-one; broadcasted masks; NaNs from missing eps; invalid id handling in embedding; odd nibble in INT4; using tl.math.sin.
  - Fix: stable math (online max-sub softmax / eps in norms), correct pointer math, consistent casts, guard tails, int4 ops on int32, use tl.sin.

- Performance outlook (post-fix, score-aware)
  - GEMM-like: BM,BN∈{64,128,256}, BK∈{16,32,64}, GROUP_M∈{4,8}, warps∈{4,8,16}, stages∈{1,2,3}.
  - Elementwise/reduction: BLOCK_SIZE∈{256,512,1024}, warps∈{2,4,8}, stages=1.
  - Add tl.multiple_of / tl.max_contiguous on inner-dim pointers when valid.
"""

prompt_ga = r"""
You are an expert in Triton performance on **AMD ROCm**. Analyze the code and numbers; explain the optimization strategy, classify bottlenecks, and propose actions to beat the **Golden Reference** without renaming functions.

Original problem:
{problem}

Triton code:
{code}

Test results:
latency: {latency}
efficiency (TFLOPS / GB/s): {efficiency}

Return only a code block with tag reflection including:

- What it does well
  - Tiling/dataflow, memory coalescing, fp32 accumulation, useful fusion.

- Bottleneck classification
  - Memory-bound vs compute-bound (roofline thinking).
  - Register pressure/spills; occupancy (warps, resident blocks); L2 locality (GROUP_M).
  - Grid underutilization or tail inefficiency.

- Targeted improvements (score-aware, ROCm priors)
  - GEMM-like: BM,BN∈{64,128,256}, BK∈{16,32,64}, GROUP_M∈{4,8}, warps∈{4,8,16}, stages∈{1,2,3}.
  - Elementwise/reduction: BLOCK_SIZE∈{256,512,1024}, warps∈{2,4,8}, stages=1.
  - Vectorize IO on inner contiguous dim; add tl.multiple_of/tl.max_contiguous when provably true.
  - Fuse cheap epilogues (bias/activation) if allowed; prefer online algorithms for softmax/LayerNorm.
"""

prompt_rocm = r"""
You are an expert in Triton operators for **AMD ROCm**. Diagnose failed tests with MI-series Wave64 realities and autotune best practices.

Original problem:
{problem}

Attempted solution:
{solution}

Test results:
{test_result}

Return only a code block with tag reflection listing:
- Root causes with precise symptoms (grid rank, constexpr, masks, strides, dtypes, fp32 acc, eps, loop legality).
- Minimal, precise fixes (edits at the right lines).
- Next-step tuning knobs for MI-series (SAFE→AGGRESSIVE ladder + meaningful autotune key).

Checklist:
1) PIDs vs grid rank; tl.arange constexpr; use `tl.static_range` to bound loops when needed.
2) Pointers & masks: correct strides; mask rank == tile; tl.int32 indices.
3) Dtypes & numerics: tl.dot fp16/bf16 + fp32 acc; eps; online max-sub; **tl.sin** instead of tl.math.sin.
4) Autotune: multiple configs; ladder; BLOCK_* as tl.constexpr; include GROUP_M for L2 reuse.
5) Wave64 perf: warps∈{2,4,8,16}; tiles multiples of 16.

### Personalized Debugging Cards (10 tasks)

- **FLASH_DECODE2_PHI**: Replace runtime loop with `tl.static_range(SEQ_BLK_MAX)` + mask；fp32 accumulators；`sum_exp==0` → write zeros；grid(batch,head)；vectorize head_dim。
- **L2_NORM_FWD**: Independent strides；zero masked values；`var=sum(x*x)` fp32；`rstd = 1/sqrt(var+eps)`；mask `cols<N`。
- **L2_NORM_BWD**: Use `dx=(dy - y*sum(dy*y))*rstd`；all inner sums fp32；mask tails；respect strides。
- **INT4_MATMUL**: Cast packed/zp to int32 before shifts；`BLOCK_K%8==0`；per-K-group scale/zp；SPLIT_K>1 uses atomic_add；mask K/N tails；avoid modulo wrap。
- **SIN_KERNEL**: Switch to **tl.sin**；1D grid；BLOCK_SIZE 512/1024；vectorize contiguous inner dim。
- **TRITON_MATMUL**: 2D tiles + GROUP_M；fp32 acc；K-tail masks；avoid fp8 unless required；SAFE→AGGR configs。
- **MATRIX_TRANSPOSE**: Use 2D grid with tiles T=32/64；mask both edges；coalesce inner dim；do not launch with grid=(1,)。
- **EMBEDDING**: Follow signature exactly；mask OOB ids to 0；coalesce along hidden；BLOCK_DMODEL=pow2；warps 1–2。
- **ROTARY_TRANSFORM**: Handle interleaved & non-interleaved；varlen via cu_seqlens；cos/sin/x in fp32；preserve tail if rotary_dim<headdim。
- **MATRIX_VECTOR_MULTIP**: Parallelize over N；stream K with BK；preload B per chunk；fp32 acc；coalesce A[K]；warps 4/8，stages 2。
"""

prompt_exe_rocm = r"""
You are an expert in Triton on **AMD ROCm**. Two tests were run (runnable vs correctness). Diagnose and fix **without** renaming functions.

Original problem:
{problem}

Attempted solution:
{solution}

Results for runnable test:
{call_test_result}

Results for correctness test:
{exe_test_result}

Return only a code block with tag reflection:

- Runnable issues
  - Grid vs PIDs; tl.arange constexpr; wrong mask rank; pointer/stride bugs; passing BLOCK_* from wrapper; CUDA-only uses; missing out allocation; dynamic loop range.
  - Fixes: grid tuple correction; tl.int32 casts; mask ranks；pointer math fixes；remove CUDA bits；make BLOCK_* tl.constexpr；allocate `out` when None；replace dynamic loop with `tl.static_range(UPPER)` + mask.

- Correctness issues
  - Reduction axes；dtype casts；fp32 acc；tail masks；tl.dot tile ranks；invalid-id mask (embedding)；odd-nibble guard (int4)；`tl.math.sin` misuse.
  - Fixes: add eps/online-stable equations；pointer fixes & consistent casts；OOB masks；int4 shifts on int32；use tl.sin.

- Immediate ROCm autotune set (score-aware)
  - GEMM-like: BM,BN∈{64,128,256}, BK∈{16,32,64}, GROUP_M∈{4,8}, warps∈{4,8,16}, stages∈{1,2,3}.
  - Reduction/elementwise: BLOCK_SIZE∈{256,512,1024}, warps∈{2,4,8}, stages=1.
"""