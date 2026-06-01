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

// GPU-only 模式：仅构建网格，跳过 CPU 端的 mask 分配和 CPU 抽象计算
// 所有 mask 计算在 GPU 上执行，最小化主机到设备的数据传输
// 
// 优化效果：
//   - 删除 CPU 端 candidate_mask 分配（节省 ~200ms）
//   - 删除 CPU 端 valid_mask 分配（节省 ~200ms）
//   - 总计节省 ~400ms 的 CPU 端初始化开销
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

    // 优化：不在 CPU 端分配 candidate_mask 和 valid_mask
    // 这两个数组将直接在 GPU 上计算，避免 ~400ms 的 CPU 端初始化开销
    // prepared.candidate_mask 和 prepared.valid_mask 保持为空
    std::cout << "GPU-minimal: Skipping CPU-side mask allocation (will compute on GPU)" << std::endl;

    // 跳过 CPU 抽象计算 - GPU 将直接计算
    prepared.abstraction.pair_count = prepared.state_grid.total_size * prepared.input_grid.total_size;
    prepared.abstraction_ms = 0.0;  // CPU 未执行抽象
    std::cout << "GPU-minimal: Skipping CPU abstraction (GPU will compute directly)" << std::endl;
    
    prepared.budget = estimate_memory_budget(prepared.state_grid, prepared.input_grid);
    return prepared;
}

}  // namespace gsc
