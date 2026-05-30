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
inline std::vector<std::uint8_t> build_valid_mask(const Grid4D<StateIndex>& grid,
                                                  const CaseConfig& cfg) {
    std::vector<std::uint8_t> mask(grid.total_size, 0);
    for (std::uint64_t flat = 0; flat < grid.total_size; ++flat) {
        double center[kStateDim];
        grid.center(static_cast<StateIndex>(flat), center);
        
        // 检查是否在 map 内
        bool valid = point_in_rect(cfg.map, center);
        
        // 检查是否在障碍物内
        if (valid) {
            for (const auto& obstacle : cfg.obstacles) {
                if (point_in_rect(obstacle, center)) {
                    valid = false;
                    break;
                }
            }
        }
        
        // 检查是否满足状态约束（如双曲线约束）
        if (valid && cfg.constraint_type != ConstraintType::kNone) {
            valid = satisfies_state_constraint(cfg, center);
        }
        
        mask[flat] = valid ? 1 : 0;
    }
    return mask;
}

// 检查索引盒内的所有状态是否都满足状态约束
// 返回 true 表示所有状态都满足约束
// 注意：此函数仅用于 CPU 版本，GPU 版本使用前缀和快速检查
template <typename StateIndex>
inline bool box_satisfies_constraint(const Grid4D<StateIndex>& grid,
                                     const IndexBox4D& box,
                                     const CaseConfig& cfg) {
    if (cfg.constraint_type == ConstraintType::kNone) {
        return true;  // 无约束，直接返回 true
    }
    
    // 处理 wrap-around 情况
    const int wrap_dim = grid.wrap_dim;
    const bool has_wrap = (box.min[wrap_dim] > box.max[wrap_dim]);
    
    // 遍历盒内所有状态
    std::uint32_t idx[kStateDim];
    for (idx[3] = box.min[3]; idx[3] <= box.max[3]; ++idx[3]) {
        for (idx[2] = box.min[2]; ; ++idx[2]) {
            // 处理 wrap-around：如果 wrap_dim == 2 且有环绕
            bool in_range_2 = false;
            if (wrap_dim == 2 && has_wrap) {
                in_range_2 = (idx[2] >= box.min[2] || idx[2] <= box.max[2]);
            } else {
                in_range_2 = (idx[2] <= box.max[2]);
            }
            if (!in_range_2 && idx[2] > box.max[2] && (!has_wrap || idx[2] >= grid.shape[2])) {
                break;
            }
            
            for (idx[1] = box.min[1]; idx[1] <= box.max[1]; ++idx[1]) {
                for (idx[0] = box.min[0]; idx[0] <= box.max[0]; ++idx[0]) {
                    double center[kStateDim];
                    StateIndex flat = grid.flatten(idx);
                    grid.center(flat, center);
                    
                    if (!satisfies_state_constraint(cfg, center)) {
                        return false;  // 有状态不满足约束
                    }
                }
            }
            
            // wrap-around 处理：从 max 跳到 0
            if (wrap_dim == 2 && has_wrap && idx[2] == grid.shape[2] - 1) {
                idx[2] = static_cast<std::uint32_t>(-1);  // 下一轮 ++idx[2] 变成 0
            } else if (idx[2] == box.max[2]) {
                break;
            }
        }
    }
    
    return true;
}

template <typename StateIndex>
inline PairAbstraction<StateIndex> build_abstraction_cpu(const Grid4D<StateIndex>& state_grid,
                                                         const InputGrid2D& input_grid,
                                                         const Unicycle4DModel& model,
                                                         const CaseConfig& cfg,
                                                         const std::vector<std::uint8_t>& valid_mask) {
    PairAbstraction<StateIndex> abstraction;
    abstraction.pair_count = state_grid.total_size * input_grid.total_size;
    abstraction.min_flat.assign(abstraction.pair_count, 0);
    abstraction.max_flat.assign(abstraction.pair_count, 0);
    abstraction.valid.assign(abstraction.pair_count, 0);

    // 构建约束掩码和前缀和
    std::vector<std::uint8_t> constraint_mask(state_grid.total_size, 0);
    for (std::uint64_t flat = 0; flat < state_grid.total_size; ++flat) {
        double center[kStateDim];
        state_grid.center(static_cast<StateIndex>(flat), center);
        constraint_mask[flat] = satisfies_state_constraint(cfg, center) ? 1 : 0;
    }
    
    // 构建前缀和
    auto prefix_constraint = build_prefix_sum_cpu(state_grid, constraint_mask);
    auto prefix_valid = build_prefix_sum_cpu(state_grid, valid_mask);

    for (std::uint64_t state = 0; state < state_grid.total_size; ++state) {
        double x[kStateDim];
        state_grid.center(static_cast<StateIndex>(state), x);
        
        // 检查当前状态是否满足约束
        if (cfg.constraint_type != ConstraintType::kNone) {
            if (!satisfies_state_constraint(cfg, x)) {
                continue;  // 当前状态不满足约束，跳过所有输入
            }
        }
        
        for (std::uint64_t input = 0; input < input_grid.total_size; ++input) {
            double u[kInputDim];
            input_grid.center(static_cast<InputIndex>(input), u);

            double min_coord[kStateDim];
            double max_coord[kStateDim];
            model.successor_box(x, u, state_grid.eta, min_coord, max_coord);

            IndexBox4D box{};
            const auto pair = state * input_grid.total_size + input;
            if (!continuous_box_to_index_box(state_grid, min_coord, max_coord, &box)) {
                continue;  // 后继盒越界
            }

            // 使用前缀和快速检查后继盒内所有状态是否都满足约束
            // 算法：计算盒内满足约束的状态数和有效状态数，两者相等则所有有效状态都满足约束
            const PrefixCount constraint_count = query_box_count(state_grid, prefix_constraint,
                                                                 state_grid.flatten(box.min),
                                                                 state_grid.flatten(box.max));
            const PrefixCount valid_count = query_box_count(state_grid, prefix_valid,
                                                            state_grid.flatten(box.min),
                                                            state_grid.flatten(box.max));
            
            if (constraint_count != valid_count) {
                continue;  // 有有效状态不满足约束，跳过这个 (x,u) 对
            }

            abstraction.min_flat[pair] = state_grid.flatten(box.min);
            abstraction.max_flat[pair] = state_grid.flatten(box.max);
            abstraction.valid[pair] = 1;
        }
    }

    return abstraction;
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

    // CPU 版本：构建 candidate mask（不涉及约束检查）
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

template <typename StateIndex>
inline PreparedCase<StateIndex> prepare_case_cpu(const CaseConfig& cfg) {
    PreparedCase<StateIndex> prepared;
    prepared.state_grid = build_state_grid<StateIndex>(cfg.state_lb, cfg.state_ub, cfg.state_eta, cfg.wrap_dim);
    prepared.input_grid = build_input_grid(cfg.input_lb, cfg.input_ub, cfg.input_eta);
    prepared.model.sample_time = cfg.sample_time;
    prepared.model.integration_substeps = cfg.integration_substeps;
    for (int dim = 0; dim < kStateDim; ++dim) {
        prepared.model.disturbance_half_width[dim] = cfg.disturbance_half_width[dim];
    }

    std::cout << "CPU: Building valid mask for " << prepared.state_grid.total_size << " states..." << std::endl;
    prepared.valid_mask = build_valid_mask(prepared.state_grid, cfg);
    std::cout << "CPU: Building candidate mask for " << prepared.state_grid.total_size << " states..." << std::endl;
    prepared.candidate_mask = build_candidate_mask(prepared.state_grid, cfg.candidate);
    for (std::uint64_t i = 0; i < prepared.candidate_mask.size(); ++i) {
        if (prepared.candidate_mask[i] && !prepared.valid_mask[i]) {
            throw std::runtime_error("candidate set contains invalid states");
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    std::cout << "CPU: Starting abstraction phase for " << prepared.state_grid.total_size << " states and "
              << prepared.input_grid.total_size << " inputs..." << std::endl;
    prepared.abstraction = build_abstraction_cpu(prepared.state_grid, prepared.input_grid, prepared.model, cfg, prepared.valid_mask);
    auto t1 = std::chrono::steady_clock::now();
    prepared.abstraction_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU: Abstraction phase completed in " << prepared.abstraction_ms << " ms" << std::endl;
    std::cout << "CPU: Valid pairs: " << prepared.abstraction.pair_count << std::endl;
    prepared.budget = estimate_memory_budget(prepared.state_grid, prepared.input_grid);
    return prepared;
}

}  // namespace gsc
