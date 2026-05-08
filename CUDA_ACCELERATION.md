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

GPU mode collects detailed timing data in `GpuRunReport`:

```cpp
struct GpuRunReport {
    std::vector<IterationStats> iteration_stats;
    KernelTimings kernel_timings;
    Grid4D<std::uint32_t> state_grid;
    InputGrid2D input_grid;
    MemoryBudget budget;
    std::uint64_t pair_count = 0;
};
```

### Kernel-Level Timings

```cpp
struct KernelTimings {
    double abstraction_kernel_ms;
    double scatter_mask_ms;
    double scan_axis0_ms, scan_axis1_ms, scan_axis2_ms, scan_axis3_ms;
    double pair_satisfaction_ms;
    double reduce_inputs_ms;
    double memcpy_h2d_ms, memcpy_d2h_ms;
};
```

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

## GPU Performance Characteristics

| Problem Size | Memory Usage | Abstraction Time |
|--------------|--------------|------------------|
| 10^6 pairs | ~50 MB | 10-100 ms |
| 10^8 pairs | ~5 GB | 1-10 s |
| 10^9 pairs | ~50 GB | 10-100 s |

See `ALGORITHM.md` for memory estimation formulas.