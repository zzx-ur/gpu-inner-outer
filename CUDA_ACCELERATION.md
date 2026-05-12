# CUDA Acceleration Documentation

## Overview

The project uses CUDA to parallelize grid symbolic control computations on NVIDIA GPUs.

## Core GPU Kernels

### 1. `abstraction_kernel`

Computes successor boxes for all (state, input) pairs in parallel.

**Input:**
- `state_grid` - 4D state discretization
- `input_grid` - 2D input discretization
- `model` - System dynamics model

**Output:**
- `pair_min_flat[p]` - flattened minimum index of successor box
- `pair_max_flat[p]` - flattened maximum index of successor box
- `pair_valid[p]` - validity of each (state, input) pair

### 2. `scatter_mask_kernel` + `scan_axis{0,1,2,3}_kernel`

Builds padded prefix sums for box queries with wrap-around handling.

**Input:** `mask[states]`, prefix layout

**Output:** `prefix[(n0+1)(n1+1)(n2+1)(n3+1)]`

### 3. `pair_satisfaction_kernel`

Checks which (state, input) pairs satisfy the inclusion condition.

**Input:**
- `pair_min_flat[p]`, `pair_max_flat[p]`, `pair_valid[p]`
- `prefix_reachable`, `prefix_valid`

**Output:** `pair_satisfied[p]`

### 4. `reduce_inputs_kernel`

Reduces satisfied inputs to produce controller and reachability update.

**Output:**
- `reachable_next[q]`
- `controller[q]`
- `reach_step[q]`
- `new_reachable`, `new_candidate_certified`

## GPU-Only Optimizations

### Minimal Preparation Path

`prepare_case_gpu_minimal<T>()` prepares only grid and masks, skipping CPU abstraction computation.

```cpp
template <typename StateIndex>
inline PreparedCase<StateIndex> prepare_case_gpu_minimal(const CaseConfig& cfg) {
    // 构建网格、模型、valid_mask、candidate_mask
    // 跳过 build_abstraction_cpu()
    prepared.abstraction.pair_count = prepared.state_grid.total_size * prepared.input_grid.total_size;
    prepared.abstraction_ms = 0.0;
    return prepared;
}
```

**Performance gains:**
- `--gpu` mode: skips 2 CPU abstractions → ~10-100ms saved for small cases, ~minutes for large cases
- `--compare` mode: skips 1 CPU abstraction

## Performance Data Collection

GPU mode collects comprehensive timing and memory data in `GpuRunReport`:

```cpp
struct GpuRunReport {
    // ... other fields ...
    double total_ms = 0.0;                          // 总耗时（abstraction + solve）
    KernelTimings kernel_timings;
    GpuMemoryStats memory_stats;
    std::vector<IterationStats> iteration_stats;
};
```

### Kernel-Level Timings

```cpp
struct KernelTimings {
    // Abstraction阶段计时
    double abstraction_kernel_ms = 0.0;             // 抽象kernel耗时
    double abstraction_memcpy_h2d_ms = 0.0;         // abstraction阶段host到device传输耗时
    
    // 前缀和构建计时
    double scatter_mask_ms = 0.0;                   // scatter mask耗时
    double scan_axis0_ms = 0.0;                     // axis0扫描耗时
    double scan_axis1_ms = 0.0;                     // axis1扫描耗时
    double scan_axis2_ms = 0.0;                     // axis2扫描耗时
    double scan_axis3_ms = 0.0;                     // axis3扫描耗时
    double prefix_build_total_ms = 0.0;             // 前缀和构建总耗时（scatter+scan）
    
    // 可达集合迭代计时
    double pair_satisfaction_ms = 0.0;              // pair satisfaction耗时
    double reduce_inputs_ms = 0.0;                  // reduce inputs耗时
    double iteration_memcpy_ms = 0.0;               // 迭代中memcpy耗时
    
    // 结果复制计时
    double result_memcpy_d2h_ms = 0.0;              // 结果从device到host传输耗时
    
    // 总计时
    double total_kernel_ms = 0.0;                   // 所有kernel总耗时
    double total_memcpy_ms = 0.0;                   // 所有memcpy总耗时
    double total_compute_ms = 0.0;                  // 所有计算总耗时（kernel+memcpy）
};
```

### GPU Memory Statistics

```cpp
struct GpuMemoryStats {
    std::uint64_t total_allocated_bytes = 0;        // 总分配显存
    std::uint64_t peak_usage_bytes = 0;             // 峰值显存使用
    std::uint64_t abstraction_bytes = 0;            // abstraction阶段显存
    std::uint64_t prefix_bytes = 0;                 // 前缀和显存
    std::uint64_t iteration_bytes = 0;              // 迭代阶段显存
    std::uint64_t result_bytes = 0;                 // 结果存储显存
};
```

### Per-Iteration Statistics

```cpp
struct IterationStats {
    std::uint32_t iteration = 0;                    // 迭代编号
    std::uint64_t newly_reachable = 0;              // 本次迭代新增可达状态数
    std::uint64_t newly_certified = 0;              // 本次迭代新增认证候选数
    std::uint64_t total_reachable = 0;              // 累计可达状态数
    std::uint64_t total_certified = 0;              // 累计认证候选数
    double iteration_ms = 0.0;                      // 本次迭代总耗时
    double prefix_build_ms = 0.0;                   // 前缀和构建耗时
    double satisfaction_check_ms = 0.0;             // 满足性检查耗时
    double reduction_ms = 0.0;                      // 输入归约耗时
};
```

### Timing Breakdown

**Abstraction Phase:**
- `abstraction_kernel_ms` - GPU kernel execution time for computing successor boxes
- `abstraction_memcpy_h2d_ms` - Host-to-device memory transfer for initialization

**Reachability Iteration Phase (per iteration):**
- `prefix_build_ms` - Prefix sum construction (scatter + 4 scan kernels)
- `satisfaction_check_ms` - Pair satisfaction kernel execution
- `reduction_ms` - Input reduction kernel execution + result memcpy

**Result Copy Phase:**
- `result_memcpy_d2h_ms` - Device-to-host transfer of all results and abstraction data

**Summary Metrics:**
- `total_kernel_ms` - Sum of all kernel execution times
- `total_memcpy_ms` - Sum of all memory transfer times
- `total_compute_ms` - Total kernel + memcpy time
- `total_ms` - Overall elapsed time (abstraction + solve)

## Memory Layout

### Abstraction Storage
- `min_flat`: 4 bytes × pair_count
- `max_flat`: 4 bytes × pair_count
- `valid`: 1 byte × pair_count

### Prefix Sum Layout
- Padded array size: `(n0+1) × (n1+1) × (n2+1) × (n3+1)` elements
- Each element: 16 bytes (four 4-byte counters)

## Wrap-Around Handling in GPU

For the angle dimension (wrap_dim = 2):
- Abstraction allows theta boxes to cross `[-π, π)` boundary
- Storage keeps single `min_flat` and `max_flat` values
- Box query splits wrap-around boxes into two segments

## CUDA Requirements

- NVIDIA GPU with CUDA compute capability 6.0+
- CUDA Toolkit 11.0+
- Proper GPU memory (typically 8GB+ for large problems)

## Building with CUDA

```bash
cmake -S . -B build -DGSC_ENABLE_CUDA=ON
cmake --build build -j 8
```

## Usage and Output

### Running GPU Mode

```bash
./build/cuda_symbolic_control_cli config.cfg --gpu
```

### Output Format

The GPU mode outputs comprehensive timing and memory information:

```
[GPU Detailed Timing]
=== Abstraction Phase ===
  Kernel execution      : 123.45 ms
  H2D memcpy            : 12.34 ms
  Total abstraction     : 135.79 ms

=== Reachability Iteration Phase ===
  Prefix build (total)  : 456.78 ms
  Pair satisfaction     : 234.56 ms
  Reduce inputs         : 123.45 ms
  Iteration memcpy      : 45.67 ms
  Total solve           : 860.46 ms

=== Result Copy Phase ===
  D2H memcpy            : 89.01 ms

=== Summary ===
  Total kernel time     : 938.24 ms
  Total memcpy time     : 147.02 ms
  Total compute time    : 1085.26 ms
  Total elapsed time    : 1221.05 ms

=== GPU Memory Usage ===
  Abstraction data      : 256.00 MB
  Prefix sum data       : 512.00 MB
  Iteration data        : 128.00 MB
  Total allocated       : 896.00 MB
  Peak usage            : 896.00 MB

=== Per-Iteration Breakdown (first 5 and last 5) ===
  Iter 1: 123.45 ms total [prefix: 45.67 ms, satisfaction: 34.56 ms, reduction: 23.45 ms] -> 1234 new reachable, 567 new certified
  Iter 2: 120.12 ms total [prefix: 44.33 ms, satisfaction: 33.22 ms, reduction: 22.11 ms] -> 890 new reachable, 456 new certified
  ...
  Iter 10: 98.76 ms total [prefix: 35.43 ms, satisfaction: 28.90 ms, reduction: 18.76 ms] -> 123 new reachable, 45 new certified
```

### Accessing Timing Data Programmatically

```cpp
const auto gpu = gsc::run_case_cuda_u32(cfg);

// Access overall timing
double total_time = gpu.total_ms;
double abstraction_time = gpu.abstraction_ms;
double solve_time = gpu.solve_ms;

// Access kernel-level timing
double kernel_time = gpu.kernel_timings.abstraction_kernel_ms;
double memcpy_time = gpu.kernel_timings.total_memcpy_ms;
double compute_time = gpu.kernel_timings.total_compute_ms;

// Access memory statistics
uint64_t peak_memory = gpu.memory_stats.peak_usage_bytes;
uint64_t total_allocated = gpu.memory_stats.total_allocated_bytes;

// Access per-iteration statistics
for (const auto& iter_stat : gpu.iteration_stats) {
    std::cout << "Iteration " << iter_stat.iteration 
              << ": " << iter_stat.iteration_ms << " ms"
              << " (prefix: " << iter_stat.prefix_build_ms << " ms)"
              << " (satisfaction: " << iter_stat.satisfaction_check_ms << " ms)"
              << " (reduction: " << iter_stat.reduction_ms << " ms)"
              << std::endl;
}
```

### HDF5 Output

All timing and memory data is saved to the HDF5 output file under the `/gpu/kernel_timings` and `/gpu/memory_stats` groups for further analysis.
