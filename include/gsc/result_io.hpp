#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gsc/abstraction.hpp"
#include "gsc/config.hpp"
#include "gsc/reachability.hpp"

namespace gsc {

// 前向声明
struct GpuRunReport;
struct IterationStats;

struct ResultExportView {
    std::string label;
    const SolveResult<std::uint32_t>* result = nullptr;
};

bool hdf5_export_compiled();

std::string default_result_path(const CaseConfig& cfg, const std::string& mode);

void write_results_hdf5(const std::string& output_path,
                        const CaseConfig& cfg,
                        const PreparedCase<std::uint32_t>& prepared,
                        const std::vector<ResultExportView>& runs);

// GPU专用：写入包含抽象数据和逐迭代统计的完整结果
void write_gpu_results_hdf5(const std::string& output_path,
                            const CaseConfig& cfg,
                            const GpuRunReport& gpu_report);

}  // namespace gsc
