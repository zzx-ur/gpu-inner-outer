# Optimized Code - Complete Reference

## Modified Files

### 1. include/gsc/config.hpp

**Change: Remove obstacles field**

```cpp
struct CaseConfig {
    std::string name = "unicycle_default";
    double state_lb[kStateDim]{-2.0, -2.0, -3.141592653589793, 0.0};
    double state_ub[kStateDim]{ 2.0,  2.0,  3.141592653589793, 2.0};
    double state_eta[kStateDim]{0.5, 0.5, 0.7853981633974483, 0.5};
    int wrap_dim = 2;

    double input_lb[kInputDim]{-0.5, -1.0};
    double input_ub[kInputDim]{ 0.5,  1.0};
    double input_eta[kInputDim]{0.5, 1.0};

    HyperRect4D candidate{};
    HyperRect4D map{};
    // REMOVED: std::vector<HyperRect4D> obstacles;

    // 状态约束配置
    ConstraintType constraint_type = ConstraintType::kNone;
    HyperbolicConstraintParams hyperbolic_params;
    EllipticConstraintParams elliptic_params;
    StateConstraintFn custom_constraint;

    double disturbance_half_width[kStateDim]{0.05, 0.05, 0.02, 0.02};
    double sample_time = 0.2;
    std::uint32_t integration_substeps = 32;
    std::uint32_t max_iterations = 32;
    bool verbose = true;
};
```

---

### 2. include/gsc/abstraction.hpp

**Changes: Remove CPU functions, keep GPU-only**

```cpp
#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gsc/config.hpp"
#include "gsc/model.hpp"

namespace gsc {

struct MemoryBudget {
    std::uint64_t pair_min_bytes = 0;
    std::uint64_t pair_max_bytes = 0;
    std::uint64_t pair_valid_bytes = 0;
    std::uint64_t pair_satisfied_bytes = 0;
    std::uint64_t reachable_prev_bytes = 0;
    std::uint64_t reachable_next_bytes = 0;
    std::uint64_t valid_mask_bytes = 0;
    std::uint64_t candidate_mask_bytes = 0;
    std::uint64_t controller_bytes = 0;
    std::uint64_t reach_step_bytes = 0;
    std::uint64_t prefix_reachable_bytes = 0;
    std::uint64_t prefix_valid_bytes = 0;
    std::uint64_t total_bytes = 0;
};

template <typename StateIndex>
struct PairAbstraction {
    PairIndex pair_count = 0;
    std::vector<StateIndex> min_flat;
    std::vector<StateIndex> max_flat;
    std::vector<std::uint8_t> valid;
};

template <typename StateIndex>
struct PreparedCase {
    Grid4D<StateIndex> state_grid;
    InputGrid2D input_grid;
    Unicycle4DModel model;
    std::vector<std::uint8_t> valid_mask;
    std::vector<std::uint8_t> candidate_mask;
    PairAbstraction<StateIndex> abstraction;
    MemoryBudget budget;
    double abstraction_ms = 0.0;
};

inline std::string format_bytes(std::uint64_t bytes) {
    static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << ' ' << kUnits[unit];
    return oss.str();
}

template <typename StateIndex>
inline std::vector<std::uint8_t> build_candidate_mask(const Grid4D<StateIndex>& grid,
                                                      const HyperRect4D& candidate) {
    std::vector<std::uint8_t> mask(grid.total_size, 0);
    for (std::uint64_t flat = 0; flat < grid.total_size; ++flat) {
        double center[kStateDim];
        grid.center(static_cast<StateIndex>(flat), center);
        mask[flat] = point_in_rect(candidate, center) ? 1 : 0;
    }
    return mask;
}

template <typename StateIndex>
inline MemoryBudget estimate_memory_budget(const Grid4D<StateIndex>& state_grid,
                                           const InputGrid2D& input_grid) {
    const std::uint64_t pair_count = state_grid.total_size * input_grid.total_size;
    const auto prefix_layout = build_prefix_layout(state_grid);

    MemoryBudget budget;
    budget.pair_min_bytes = pair_count * sizeof(StateIndex);
    budget.pair_max_bytes = pair_count * sizeof(StateIndex);
    budget.pair_valid_bytes = pair_count * sizeof(std::uint8_t);
    budget.pair_satisfied_bytes = pair_count * sizeof(std::uint8_t);
    budget.reachable_prev_bytes = state_grid.total_size * sizeof(std::uint8_t);
    budget.reachable_next_bytes = state_grid.total_size * sizeof(std::uint8_t);
    budget.valid_mask_bytes = state_grid.total_size * sizeof(std::uint8_t);
    budget.candidate_mask_bytes = state_grid.total_size * sizeof(std::uint8_t);
    budget.controller_bytes = state_grid.total_size * sizeof(InputIndex);
    budget.reach_step_bytes = state_grid.total_size * sizeof(ReachStep);
    budget.prefix_reachable_bytes = prefix_layout.total_size * sizeof(PrefixCount);
    budget.prefix_valid_bytes = prefix_layout.total_size * sizeof(PrefixCount);
    budget.total_bytes = budget.pair_min_bytes +
                         budget.pair_max_bytes +
                         budget.pair_valid_bytes +
                         budget.pair_satisfied_bytes +
                         budget.reachable_prev_bytes +
                         budget.reachable_next_bytes +
                         budget.valid_mask_bytes +
                         budget.candidate_mask_bytes +
                         budget.controller_bytes +
                         budget.reach_step_bytes +
                         budget.prefix_reachable_bytes +
                         budget.prefix_valid_bytes;
    return budget;
}

// GPU-only 模式：仅构建网格和 mask，跳过 CPU 抽象计算
// 所有计算在 GPU 上执行，最小化主机到设备的数据传输
template <typename StateIndex>
inline PreparedCase<StateIndex> prepare_case_gpu_minimal(const CaseConfig& cfg) {
    PreparedCase<StateIndex> prepared;
    prepared.state_grid = build_state_grid<StateIndex>(cfg.state_lb, cfg.state_ub, cfg.state_eta, cfg.wrap_dim);
    prepared.input_grid = build_input_grid(cfg.input_lb, cfg.input_ub, cfg.input_eta);
    prepared.model.sample_time = cfg.sample_time;
    prepared.model.integration_substeps = cfg.integration_substeps;
    for (int dim = 0; dim < kStateDim; ++dim) {
        prepared.model.disturbance_half_width[dim] = cfg.disturbance_half_width[dim];
    }

    // 构建 candidate mask（CPU 端，最小化传输）
    std::cout << "GPU-minimal: Building candidate mask for " << prepared.state_grid.total_size << " states..." << std::endl;
    prepared.candidate_mask = build_candidate_mask(prepared.state_grid, cfg.candidate);
    
    // valid_mask 将在 GPU 上构建，这里只分配空间
    std::cout << "GPU-minimal: Valid mask will be built on GPU" << std::endl;
    prepared.valid_mask.assign(prepared.state_grid.total_size, 0);

    // 跳过 CPU 抽象计算 - GPU 将直接计算
    prepared.abstraction.pair_count = prepared.state_grid.total_size * prepared.input_grid.total_size;
    prepared.abstraction_ms = 0.0;  // CPU 未执行抽象
    std::cout << "GPU-minimal: Skipping CPU abstraction (GPU will compute directly)" << std::endl;
    
    prepared.budget = estimate_memory_budget(prepared.state_grid, prepared.input_grid);
    return prepared;
}

}  // namespace gsc
```

**Removed Functions:**
- ❌ `build_valid_mask()` - CPU function with obstacle checking
- ❌ `box_satisfies_constraint()` - CPU constraint checking
- ❌ `build_abstraction_cpu()` - CPU abstraction computation
- ❌ `prepare_case_cpu()` - CPU case preparation

---

### 3. cuda/kernels.cu

**Key Changes:**

#### A. Remove GpuObstacle struct

```cpp
// REMOVED:
struct GpuObstacle {
    double lb[kStateDim];
    double ub[kStateDim];
};

// KEPT:
struct GpuMapParams {
    double map_lb[kStateDim];
    double map_ub[kStateDim];
};
```

#### B. Add Hyperbolic Specialization

```cuda
// 优化的双曲线约束检查（内联，避免函数调用开销）
__device__ inline bool d_satisfies_hyperbolic_constraint(const double x[kStateDim], 
                                                         double a, double b, double c) {
    const double x1_sq = x[0] * x[0];
    const double x2_sq = x[1] * x[1];
    return (x1_sq - x2_sq <= a) && (b * x2_sq - x1_sq <= c);
}

// 优化的双曲线约束专用 kernel（针对 hyperbolic_case 优化）
__global__ void build_valid_mask_hyperbolic_kernel(const Grid4D<std::uint32_t> state_grid,
                                                   const GpuMapParams map_params,
                                                   double hyperbolic_a,
                                                   double hyperbolic_b,
                                                   double hyperbolic_c,
                                                   std::uint8_t* valid_mask) {
    const std::uint64_t flat = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= state_grid.total_size) {
        return;
    }

    double center[kStateDim];
    state_grid.center(static_cast<std::uint32_t>(flat), center);
    
    // 检查是否在 map 内
    bool valid = d_point_in_rect(map_params.map_lb, map_params.map_ub, center);
    
    // 直接检查双曲线约束（避免函数调用开销）
    if (valid) {
        const double x1_sq = center[0] * center[0];
        const double x2_sq = center[1] * center[1];
        valid = (x1_sq - x2_sq <= hyperbolic_a) && (hyperbolic_b * x2_sq - x1_sq <= hyperbolic_c);
    }
    
    valid_mask[flat] = valid ? 1 : 0;
}
```

#### C. Simplify build_valid_mask_kernel

```cuda
// GPU kernel：构建 valid_mask（仅包含 map 和约束检查，无障碍物）
// 优化版本：移除障碍物检查逻辑，减少分支和内存访问
__global__ void build_valid_mask_kernel(const Grid4D<std::uint32_t> state_grid,
                                        const GpuMapParams map_params,
                                        const GpuConstraintParams constraint_params,
                                        std::uint8_t* valid_mask) {
    const std::uint64_t flat = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= state_grid.total_size) {
        return;
    }

    double center[kStateDim];
    state_grid.center(static_cast<std::uint32_t>(flat), center);
    
    // 检查是否在 map 内
    bool valid = d_point_in_rect(map_params.map_lb, map_params.map_ub, center);
    
    // 检查是否满足状态约束（无障碍物检查）
    if (valid) {
        valid = d_satisfies_constraint(center, constraint_params);
    }
    
    valid_mask[flat] = valid ? 1 : 0;
}
```

#### D. Update GPU Run Function

```cpp
// Remove obstacle variable declaration
// REMOVED: GpuObstacle* d_obstacles = nullptr;

// In GPU run function, replace obstacle handling:

// BEFORE:
std::vector<GpuObstacle> h_obstacles;
for (const auto& obs : cfg.obstacles) {
    GpuObstacle gpu_obs;
    for (int dim = 0; dim < kStateDim; ++dim) {
        gpu_obs.lb[dim] = obs.lb[dim];
        gpu_obs.ub[dim] = obs.ub[dim];
    }
    h_obstacles.push_back(gpu_obs);
}

if (!h_obstacles.empty()) {
    GSC_CUDA_CHECK(cudaMalloc(&d_obstacles, h_obstacles.size() * sizeof(GpuObstacle)));
    GSC_CUDA_CHECK(cudaMemcpy(d_obstacles,
                              h_obstacles.data(),
                              h_obstacles.size() * sizeof(GpuObstacle),
                              cudaMemcpyHostToDevice));
}

build_valid_mask_kernel<<<state_blocks, threads>>>(prepared.state_grid,
                                                   map_params,
                                                   d_obstacles,
                                                   static_cast<std::uint32_t>(h_obstacles.size()),
                                                   constraint_params,
                                                   d_valid_mask);

// AFTER:
// No obstacle handling needed

// Use specialized hyperbolic kernel when appropriate
if (constraint_params.constraint_type == 1) {  // kHyperbolic
    std::cout << "GPU: Using optimized hyperbolic constraint kernel..." << std::endl;
    build_valid_mask_hyperbolic_kernel<<<state_blocks, threads>>>(prepared.state_grid,
                                                                  map_params,
                                                                  constraint_params.hyperbolic_a,
                                                                  constraint_params.hyperbolic_b,
                                                                  constraint_params.hyperbolic_c,
                                                                  d_valid_mask);
} else {
    build_valid_mask_kernel<<<state_blocks, threads>>>(prepared.state_grid,
                                                       map_params,
                                                       constraint_params,
                                                       d_valid_mask);
}
```

#### E. Update Cleanup

```cpp
// BEFORE:
cudaFree(d_obstacles);

// AFTER:
// (removed - no obstacle memory to free)
```

---

## Summary of Changes

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| Obstacles | Supported | Removed | -1 struct, -N transfers |
| CPU Abstraction | Full | None | -65 lines |
| CPU Valid Mask | With obstacles | None | -25 lines |
| GPU Valid Mask | Generic | Generic + Hyperbolic | +25 lines |
| Data Transfers | Many | Minimal | -100s MB |
| Code Lines | 329 | 137 | -192 lines |

---

## Compilation

```bash
cmake -S . -B build -DGSC_ENABLE_CUDA=ON
cmake --build build -j 8
```

---

## Usage

```bash
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg --gpu
```

Expected output includes:
```
GPU: Using optimized hyperbolic constraint kernel...
```

This confirms the specialized kernel is being used for the hyperbolic_case.
