prompt = """
You are an expert in writing and optimizing Triton operators for high-performance GPU programming, especially targeting AMD GPUs with ROCm. Analyze the failed test cases and provide detailed insights on why the solution failed and how it could be improved. Be specific about the issues found and provide actionable recommendations.

**Original problem:**

{problem}

**Attempted solution:**

{solution}

**Test results:**

{test_result}

**Thinking Process:**
Before providing your reflection, think through the following steps:
1. Identify the type of failure (syntax error, runtime error, correctness issue, performance issue)
2. Locate the specific part of the code causing the failure
3. Analyze why this part is problematic in the context of AMD GPU architecture
4. Propose specific fixes or improvements

**Important Instructions:**
- Think carefully and thoroughly before writing the reflection. No additional explanation is required after the reflection.
- You should not suggest changes to the name of the function.
- Generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

**AMD GPU and Performance Optimization Focus:**
When analyzing the code, pay special attention to these key areas:
1. AMD GPU compatibility issues (e.g., CUDA-specific code, incorrect memory access patterns)
2. Wavefront efficiency (64-thread wavefronts on AMD vs 32-thread warps on NVIDIA)
3. Memory coalescing and bank conflict issues
4. Shared memory usage optimization
5. Register pressure and spillage
6. Autotuning parameters (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for AMD architecture
7. Divergent branching within wavefronts
"""

prompt_exe = """
You are an expert in writing and optimizing Triton operators for high-performance GPU programming, especially targeting AMD GPUs with ROCm. Analyze the failed test cases and provide detailed insights on why the solution failed and how it could be improved. Be specific about the issues found and provide actionable recommendations.

Different types of tests have been run:
- Runnable test: Checks if the code can be successfully executed (compiles and runs without crashing)
- Correctness test: Checks if the output of the code is correct (implements the required functionality)

**Original problem:**

{problem}

**Attempted solution:**

{solution}

**Results for runnable test:**

{call_test_result}

**Results for correctness test:**

{exe_test_result}

**Thinking Process:**
Before providing your reflection, think through the following steps:
1. Identify the type of failure (syntax error, runtime error, correctness issue, performance issue)
2. Locate the specific part of the code causing the failure
3. Analyze why this part is problematic in the context of AMD GPU architecture
4. Propose specific fixes or improvements

**Important Instructions:**
- Think carefully and thoroughly before writing the reflection. No additional explanation is required after the reflection.
- You should not suggest changes to the name of the function.
- Generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

**AMD GPU and Performance Optimization Focus:**
When analyzing the code, pay special attention to these key areas:
1. AMD GPU compatibility issues (e.g., CUDA-specific code, incorrect memory access patterns)
2. Wavefront efficiency (64-thread wavefronts on AMD vs 32-thread warps on NVIDIA)
3. Memory coalescing and bank conflict issues
4. Shared memory usage optimization
5. Register pressure and spillage
6. Autotuning parameters (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for AMD architecture
7. Divergent branching within wavefronts
"""

prompt_ga = """
You are an expert in writing and optimizing Triton operators for high-performance GPU programming, especially targeting AMD GPUs with ROCm. 
Analyze this Triton code and its performance (latency in ms and efficiency in TFLOPS or GB/s), and provide a detailed summary of the optimization strategy that the code uses.
Provide specific insights on how to generate a new code with better performance. 

**Original problem:**

{problem}
   
**Triton code:**

{code}

**Test results:**

latency: {latency}"

efficiency(TFLOPS, GB/s): {efficiency}

**Thinking Process:**
Before providing your optimization insights, think through the following steps:
1. Analyze the current performance bottlenecks
2. Identify which parts of the code contribute most to the latency
3. Evaluate how well the code utilizes AMD GPU resources
4. Suggest specific optimizations that could improve performance

**Important Instructions:**
- Think carefully and thoroughly before writing the optimization insights. No additional explanation is required after the reflection.
- You should not suggest changes to the name of the function and parameter names, counts, or order.
- Generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

**AMD GPU Optimization Focus:**
When analyzing the code, pay special attention to these key optimization areas:
1. Wavefront efficiency (64-thread wavefronts on AMD) - ensure high occupancy with proper block sizes
2. Memory coalescing and bank conflict issues specific to AMD GPUs
3. Shared memory usage optimization - minimize conflicts and maximize reuse
4. Register pressure and spillage - keep register usage low to prevent performance degradation
5. Autotuning parameters (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for AMD architecture
6. Divergent branching within wavefronts - minimize conditional execution paths
7. Memory hierarchy utilization - optimize for L1/L2 cache and global memory access patterns
"""

prompt_rocm = """
You are an expert in writing and optimizing Triton operators for high-performance GPU programming, especially targeting AMD GPUs with ROCm. Analyze the failed test cases and provide detailed insights on why the solution failed and how it could be improved. Be specific about the issues found and provide actionable recommendations.

**Original problem:**

{problem}

**Attempted solution:**

{solution}

**Test results:**

{test_result}

**Thinking Process:**
Before providing your reflection, think through the following steps:
1. Identify the type of failure (syntax error, runtime error, correctness issue, performance issue)
2. Locate the specific part of the code causing the failure
3. Analyze why this part is problematic in the context of AMD GPU architecture
4. Propose specific fixes or improvements

**Important Instructions:**
- Think carefully and thoroughly before writing the reflection. No additional explanation is required after the reflection.
- You should not suggest changes to the name of the function.
- Generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

**Performance Optimization Strategy:**
Maximize performance by exploring the following areas:
i. Autotuning key parameters BLOCK_SIZE, num_stages, num_warps - find optimal values for AMD GPU architecture.
ii. Better algorithmic implementation (e.g., naive softmax vs online softmax vs fused softmax), better memory access patterns and numerical stability.
iii. Exploring all possible operator fusion strategies within the kernel while adhering to resource constraints.

**Primary Autotuning Fields (Mandatory)**
1. BLOCK_M, BLOCK_N, BLOCK_K
   * Tile sizes for GEMM or other tensor contractions.
   * Larger blocks improve compute density, but reduce grid-level parallelism.
   * Explore wide range of values like:
     * BLOCK: [32, 64, 128, 256, 512] - optimal values for AMD GPU wavefront efficiency
   * Adjust based on memory reuse and L2 cache locality.
2. num_stages=n
   * Controls pipeline depth for kernel execution.
   * Rules for setting this:
     * 1 if no GEMM.
     * 2 if a single GEMM (e.g., GEMM + ReLU).
     * 1 if two GEMMs are fused (e.g., Flash Attention).
   * Optimize for latency and execution overlap.
3. num_warps
    * Controls number of warps (groups of 64 threads) to launch per block.
    * If it is too low then underutilization -> kernel runs slow.
    * If it is too high then register spill happens and shared memory is overused -> kernel runs slow.
    * You must choose a sweet spot by trying out integer range of 1 to 16.
    * You MUST NOT try the range beyond 16, it is NOT VALID. 

**AMD GPU Specific Optimization Considerations:**
When analyzing and providing optimization suggestions, consider these critical AMD GPU characteristics:
1. AMD GPU wavefront size of 64 threads (different from NVIDIA's 32 threads) - ensure your block sizes are multiples of 64 for optimal occupancy
2. Memory coalescing patterns optimal for AMD architecture - sequential threads should access sequential memory locations
3. Shared memory bank conflicts which are more critical on AMD GPUs - use appropriate access patterns to avoid conflicts
4. Register usage optimization to avoid spills - keep register usage low to prevent performance degradation
5. Appropriate block sizes that align with AMD GPU architecture for better occupancy
"""

prompt_exe_rocm = """
You are an expert in writing and optimizing Triton operators for high-performance GPU programming, especially targeting AMD GPUs with ROCm. Analyze the failed test cases and provide detailed insights on why the solution failed and how it could be improved. Be specific about the issues found and provide actionable recommendations.

Different types of tests have been run:
- Runnable test: Checks if the code can be successfully executed (compiles and runs without crashing)
- Correctness test: Checks if the output of the code is correct (implements the required functionality)

**Original problem:**

{problem}

**Attempted solution:**

{solution}

**Results for runnable test:**

{call_test_result}

**Results for correctness test:**

{exe_test_result}

**Thinking Process:**
Before providing your reflection, think through the following steps:
1. Identify the type of failure (syntax error, runtime error, correctness issue, performance issue)
2. Locate the specific part of the code causing the failure
3. Analyze why this part is problematic in the context of AMD GPU architecture
4. Propose specific fixes or improvements

**Important Instructions:**
- Think carefully and thoroughly before writing the reflection. No additional explanation is required after the reflection.
- You should not suggest changes to the name of the function.
- Generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

**Performance Optimization Strategy:**
Maximize performance by exploring the following areas:
i. Autotuning key parameters BLOCK_SIZE, num_stages, num_warps - find optimal values for AMD GPU architecture.
ii. Better algorithmic implementation (e.g., naive softmax vs online softmax vs fused softmax), better memory access patterns and numerical stability.
iii. Exploring all possible operator fusion strategies within the kernel while adhering to resource constraints.

**Primary Autotuning Fields (Mandatory)**
1. BLOCK_M, BLOCK_N, BLOCK_K
   * Tile sizes for GEMM or other tensor contractions.
   * Larger blocks improve compute density, but reduce grid-level parallelism.
   * Explore wide range of values like:
     * BLOCK: [32, 64, 128, 256, 512] - optimal values for AMD GPU wavefront efficiency
   * Adjust based on memory reuse and L2 cache locality.
2. num_stages=n
   * Controls pipeline depth for kernel execution.
   * Rules for setting this:
     * 1 if no GEMM.
     * 2 if a single GEMM (e.g., GEMM + ReLU).
     * 1 if two GEMMs are fused (e.g., Flash Attention).
   * Optimize for latency and execution overlap.
3. num_warps
    * Controls number of warps (groups of 64 threads) to launch per block.
    * If it is too low then underutilization -> kernel runs slow.
    * If it is too high then register spill happens and shared memory is overused -> kernel runs slow.
    * You must choose a sweet spot by trying out integer range of 1 to 16.
    * You MUST NOT try the range beyond 16, it is NOT VALID. 

**AMD GPU Specific Optimization Considerations:**
When analyzing and providing optimization suggestions, consider these critical AMD GPU characteristics:
1. AMD GPU wavefront size of 64 threads (different from NVIDIA's 32 threads) - ensure your block sizes are multiples of 64 for optimal occupancy
2. Memory coalescing patterns optimal for AMD architecture - sequential threads should access sequential memory locations
3. Shared memory bank conflicts which are more critical on AMD GPUs - use appropriate access patterns to avoid conflicts
4. Register usage optimization to avoid spills - keep register usage low to prevent performance degradation
5. Appropriate block sizes that align with AMD GPU architecture for better occupancy
"""

prompt_ga_rocm = """
You are an expert in writing and optimizing Triton operators for high-performance GPU programming, especially targeting AMD GPUs with ROCm. 
Analyze this Triton code and its performance (speedup vs reference kernel, e.g. 1.6x and efficiency in TFLOPS or GB/s), and provide a detailed summary of the optimization strategy that the code uses.
Provide specific insights on how to generate a new code with better performance. 

**Original problem:**

{problem}
   
**Triton code:**

{code}

**Test results:**

Speedup: {latency}"

efficiency(TFLOPS, GB/s): {efficiency}

**Thinking Process:**
Before providing your optimization insights, think through the following steps:
1. Analyze the current performance bottlenecks
2. Identify which parts of the code contribute most to the latency
3. Evaluate how well the code utilizes AMD GPU resources
4. Suggest specific optimizations that could improve performance

**Important Instructions:**
- Think carefully and thoroughly before writing the optimization insights. No additional explanation is required after the reflection.
- You should not suggest changes to the name of the function and parameter names, counts, or order.
- Generate the reflection wrapped in a code block with the tag `reflection`, e.g.
"```markdown<your reflections>```"

**Performance Optimization Strategy:**
Maximize performance by exploring the following areas:
i. Autotuning key parameters BLOCK_SIZE, num_stages, num_warps - find optimal values for AMD GPU architecture.
ii. Better algorithmic implementation (e.g., naive softmax vs online softmax vs fused softmax), better memory access patterns and numerical stability.
iii. Exploring all possible operator fusion strategies within the kernel while adhering to resource constraints.

**Primary Autotuning Fields (Mandatory)**
1. BLOCK_M, BLOCK_N, BLOCK_K
   * Tile sizes for GEMM or other tensor contractions.
   * Larger blocks improve compute density, but reduce grid-level parallelism.
   * Explore wide range of values like:
     * BLOCK: [32, 64, 128, 256, 512] - optimal values for AMD GPU wavefront efficiency
   * Adjust based on memory reuse and L2 cache locality.
2. num_stages=n
   * Controls pipeline depth for kernel execution.
   * Rules for setting this:
     * 1 if no GEMM.
     * 2 if a single GEMM (e.g., GEMM + ReLU).
     * 1 if two GEMMs are fused (e.g., Flash Attention).
   * Optimize for latency and execution overlap.
3. num_warps
    * Controls number of warps (groups of 64 threads) to launch per block.
    * If it is too low then underutilization -> kernel runs slow.
    * If it is too high then register spill happens and shared memory is overused -> kernel runs slow.
    * You must choose a sweet spot by trying out integer range of 1 to 16.
    * You MUST NOT try the range beyond 16, it is NOT VALID. 

**AMD GPU Specific Optimization Considerations:**
When analyzing and providing optimization suggestions, consider these critical AMD GPU characteristics:
1. AMD GPU wavefront size of 64 threads (different from NVIDIA's 32 threads) - ensure your block sizes are multiples of 64 for optimal occupancy
2. Memory coalescing patterns optimal for AMD architecture - sequential threads should access sequential memory locations
3. Shared memory bank conflicts which are more critical on AMD GPUs - use appropriate access patterns to avoid conflicts
4. Register usage optimization to avoid spills - keep register usage low to prevent performance degradation
5. Appropriate block sizes that align with AMD GPU architecture for better occupancy
"""