#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gsc/cuda_api.hpp"
#include "gsc/result_io.hpp"

namespace {

template <typename StateIndex>
void print_summary(const gsc::Grid4D<StateIndex>& state_grid,
                   const gsc::InputGrid2D& input_grid,
                   const gsc::MemoryBudget& budget,
                   std::uint64_t pair_count,
                   const std::vector<std::uint8_t>& candidate_mask,
                   const gsc::SolveResult<StateIndex>& result,
                   double abstraction_ms,
                   const std::string& label) {
    std::uint64_t candidate_states = 0;
    for (auto v : candidate_mask) {
        candidate_states += (v != 0);
    }

    std::cout << "\n[" << label << "]\n";
    std::cout << "states        : " << state_grid.total_size << '\n';
    std::cout << "inputs        : " << input_grid.total_size << '\n';
    std::cout << "pairs         : " << pair_count << '\n';
    std::cout << "reachable     : " << result.reachable_states << '\n';
    std::cout << "candidate     : " << candidate_states << '\n';
    std::cout << "candidate ctl : " << result.certified_candidate_states << '\n';
    std::cout << "iterations    : " << result.iterations << '\n';
    std::cout << "converged     : " << (result.converged ? "true" : "false") << '\n';
    std::cout << "message       : " << result.message << '\n';
    std::cout << "abstraction   : " << abstraction_ms << " ms\n";
    std::cout << "solve         : " << result.solve_ms << " ms\n";
    std::cout << "memory total  : " << gsc::format_bytes(budget.total_bytes) << '\n';
}

void print_gpu_detailed_timing(const gsc::GpuRunReport& gpu) {
    std::cout << "\n[GPU Detailed Timing]\n";
    std::cout << "=== Abstraction Phase ===\n";
    std::cout << "  Kernel execution      : " << gpu.kernel_timings.abstraction_kernel_ms << " ms\n";
    std::cout << "  GPU init (masks+count): " << gpu.kernel_timings.abstraction_memcpy_h2d_ms << " ms\n";
    std::cout << "  Total abstraction     : " << gpu.abstraction_ms << " ms\n";
    
    std::cout << "\n=== Reachability Iteration Phase ===\n";
    std::cout << "  Prefix build (total)  : " << gpu.kernel_timings.prefix_build_total_ms << " ms\n";
    std::cout << "  Pair satisfaction     : " << gpu.kernel_timings.pair_satisfaction_ms << " ms\n";
    std::cout << "  Reduce inputs         : " << gpu.kernel_timings.reduce_inputs_ms << " ms\n";
    std::cout << "  Iteration memcpy      : " << gpu.kernel_timings.iteration_memcpy_ms << " ms\n";
    std::cout << "  Total solve           : " << gpu.solve_ms << " ms\n";
    
    std::cout << "\n=== Result Copy Phase ===\n";
    std::cout << "  D2H memcpy            : " << gpu.kernel_timings.result_memcpy_d2h_ms << " ms\n";
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "  Total kernel time     : " << gpu.kernel_timings.total_kernel_ms << " ms\n";
    std::cout << "  Total memcpy time     : " << gpu.kernel_timings.total_memcpy_ms << " ms\n";
    std::cout << "  Total compute time    : " << gpu.kernel_timings.total_compute_ms << " ms\n";
    std::cout << "  Total elapsed time    : " << gpu.total_ms << " ms\n";
    
    std::cout << "\n=== GPU Memory Usage ===\n";
    std::cout << "  Abstraction data      : " << gsc::format_bytes(gpu.memory_stats.abstraction_bytes) << "\n";
    std::cout << "  Prefix sum data       : " << gsc::format_bytes(gpu.memory_stats.prefix_bytes) << "\n";
    std::cout << "  Iteration data        : " << gsc::format_bytes(gpu.memory_stats.iteration_bytes) << "\n";
    std::cout << "  Total allocated       : " << gsc::format_bytes(gpu.memory_stats.total_allocated_bytes) << "\n";
    std::cout << "  Peak usage            : " << gsc::format_bytes(gpu.memory_stats.peak_usage_bytes) << "\n";
    
    if (!gpu.iteration_stats.empty()) {
        std::cout << "\n=== Per-Iteration Breakdown (first 5 and last 5) ===\n";
        const auto& stats = gpu.iteration_stats;
        const int show_count = std::min(5, static_cast<int>(stats.size()));
        
        for (int i = 0; i < show_count; ++i) {
            const auto& s = stats[i];
            std::cout << "  Iter " << s.iteration << ": "
                      << s.iteration_ms << " ms total"
                      << " [prefix: " << s.prefix_build_ms << " ms"
                      << ", satisfaction: " << s.satisfaction_check_ms << " ms"
                      << ", reduction: " << s.reduction_ms << " ms]"
                      << " -> " << s.newly_reachable << " new reachable, "
                      << s.newly_certified << " new certified\n";
        }
        
        if (stats.size() > 2 * show_count) {
            std::cout << "  ...\n";
        }
        
        for (int i = std::max(show_count, static_cast<int>(stats.size()) - show_count); 
             i < static_cast<int>(stats.size()); ++i) {
            const auto& s = stats[i];
            std::cout << "  Iter " << s.iteration << ": "
                      << s.iteration_ms << " ms total"
                      << " [prefix: " << s.prefix_build_ms << " ms"
                      << ", satisfaction: " << s.satisfaction_check_ms << " ms"
                      << ", reduction: " << s.reduction_ms << " ms]"
                      << " -> " << s.newly_reachable << " new reachable, "
                      << s.newly_certified << " new certified\n";
        }
    }
}

template <typename StateIndex>
void print_summary(const gsc::PreparedCase<StateIndex>& prepared,
                   const gsc::SolveResult<StateIndex>& result,
                   const std::string& label) {
    print_summary(prepared.state_grid, prepared.input_grid, prepared.budget,
                  prepared.abstraction.pair_count, prepared.candidate_mask,
                  result, prepared.abstraction_ms, label);
}

bool same_result(const gsc::SolveResult<std::uint32_t>& a,
                 const gsc::SolveResult<std::uint32_t>& b) {
    return a.valid_mask == b.valid_mask &&
           a.candidate_mask == b.candidate_mask &&
           a.reachable_mask == b.reachable_mask &&
           a.controller == b.controller &&
           a.reach_step == b.reach_step;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: gsc_cli <config.cfg> [--cpu|--gpu|--compare] [--output result.h5]\n";
        return 1;
    }

    try {
        const std::string config_path = argv[1];
        std::string mode = "--cpu";
        std::string output_path;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--cpu" || arg == "--gpu" || arg == "--compare") {
                mode = arg;
                continue;
            }
            if (arg == "--output") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--output requires a file path");
                }
                output_path = argv[++i];
                continue;
            }
            throw std::runtime_error("unknown argument: " + arg);
        }

        const auto cfg = gsc::load_case_config(config_path);
        const std::string mode_name = mode.rfind("--", 0) == 0 ? mode.substr(2) : mode;
        if (output_path.empty()) {
            output_path = gsc::default_result_path(cfg, mode_name);
        }

        if (mode == "--cpu") {
            // CPU mode not supported - using GPU computation instead
            std::cout << "Note: CPU mode not available, using GPU computation\n";
            const auto gpu = gsc::run_case_cuda_u32(cfg);
            if (!gpu.executed) {
                std::cerr << "gpu run unavailable: " << gpu.message << '\n';
                return 2;
            }
            print_summary(gpu.state_grid, gpu.input_grid, gpu.budget, gpu.pair_count,
                         gpu.result.candidate_mask, gpu.result, gpu.abstraction_ms, "gpu");
            
            print_gpu_detailed_timing(gpu);
            
            gsc::write_gpu_results_hdf5(output_path, cfg, gpu);
            std::cout << "\nresult file   : " << output_path << '\n';
            std::cout << "  - includes abstraction data (" 
                      << (gpu.pair_count * (2 * sizeof(std::uint32_t) + sizeof(std::uint8_t)) / (1024.0 * 1024.0))
                      << " MB)\n";
            std::cout << "  - includes " << gpu.iteration_stats.size() << " iteration statistics\n";
            return 0;
        }

        if (mode == "--gpu") {
            const auto gpu = gsc::run_case_cuda_u32(cfg);
            if (!gpu.executed) {
                std::cerr << "gpu run unavailable: " << gpu.message << '\n';
                return 2;
            }
            // GPU-only 模式：直接使用 GPU report 中的网格信息，避免重复 prepare_case_cpu
            print_summary(gpu.state_grid, gpu.input_grid, gpu.budget, gpu.pair_count,
                         gpu.result.candidate_mask, gpu.result, gpu.abstraction_ms, "gpu");
            
            // 打印详细的计时和显存信息
            print_gpu_detailed_timing(gpu);
            
            // 使用GPU专用的HDF5写入函数，包含抽象数据和逐迭代统计
            gsc::write_gpu_results_hdf5(output_path, cfg, gpu);
            std::cout << "\nresult file   : " << output_path << '\n';
            std::cout << "  - includes abstraction data (" 
                      << (gpu.pair_count * (2 * sizeof(std::uint32_t) + sizeof(std::uint8_t)) / (1024.0 * 1024.0))
                      << " MB)\n";
            std::cout << "  - includes " << gpu.iteration_stats.size() << " iteration statistics\n";
            return 0;
        }

        if (mode == "--compare") {
            // Compare mode: GPU-only (CPU not supported)
            const auto gpu = gsc::run_case_cuda_u32(cfg);
            if (!gpu.executed) {
                std::cerr << "gpu compare unavailable: " << gpu.message << '\n';
                return 2;
            }

            print_summary(gpu.state_grid, gpu.input_grid, gpu.budget, gpu.pair_count,
                         gpu.result.candidate_mask, gpu.result, gpu.abstraction_ms, "gpu");
            
            print_gpu_detailed_timing(gpu);
            
            gsc::write_gpu_results_hdf5(output_path, cfg, gpu);
            std::cout << "\nresult file   : " << output_path << '\n';
            std::cout << "  - includes abstraction data (" 
                      << (gpu.pair_count * (2 * sizeof(std::uint32_t) + sizeof(std::uint8_t)) / (1024.0 * 1024.0))
                      << " MB)\n";
            std::cout << "  - includes " << gpu.iteration_stats.size() << " iteration statistics\n";
            return 0;
        }

        throw std::runtime_error("unknown mode: " + mode);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
