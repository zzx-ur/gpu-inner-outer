#pragma once

#include <string>
#include <vector>

#include "gsc/reachability.hpp"

namespace gsc {

// 单次迭代的统计信息
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

// Kernel级别的计时信息
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

// GPU显存监控信息
struct GpuMemoryStats {
    std::uint64_t total_allocated_bytes = 0;        // 总分配显存
    std::uint64_t peak_usage_bytes = 0;             // 峰值显存使用
    std::uint64_t abstraction_bytes = 0;            // abstraction阶段显存
    std::uint64_t prefix_bytes = 0;                 // 前缀和显存
    std::uint64_t iteration_bytes = 0;              // 迭代阶段显存
    std::uint64_t result_bytes = 0;                 // 结果存储显存
};

struct GpuRunReport {
    bool compiled_with_cuda = false;
    bool runtime_available = false;
    bool executed = false;
    std::string message;
    double abstraction_ms = 0.0;
    double solve_ms = 0.0;
    double total_ms = 0.0;                          // 总耗时（abstraction + solve）
    SolveResult<std::uint32_t> result;
    // GPU-only 模式需要返回的网格信息（用于输出和统计）
    Grid4D<std::uint32_t> state_grid;
    InputGrid2D input_grid;
    MemoryBudget budget;
    std::uint64_t pair_count = 0;
    // 逐迭代统计信息
    std::vector<IterationStats> iteration_stats;
    // Kernel级别计时信息
    KernelTimings kernel_timings;
    // GPU显存统计信息
    GpuMemoryStats memory_stats;
    // 抽象阶段数据（后继盒信息）
    std::vector<std::uint32_t> abstraction_min_flat;  // 后继盒最小flat索引
    std::vector<std::uint32_t> abstraction_max_flat;  // 后继盒最大flat索引
    std::vector<std::uint8_t> abstraction_valid;      // 状态-输入对有效性
};

bool cuda_backend_compiled();
GpuRunReport run_case_cuda_u32(const CaseConfig& cfg);

}  // namespace gsc
