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
        std::cerr << "usage: gsc_cli <config.cfg> [--cpu|--gpu|--gpu-contracted|--compare] [--output result.h5]\n";
        return 1;
    }

    try {
        const std::string config_path = argv[1];
        std::string mode = "--cpu";
        std::string output_path;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--cpu" || arg == "--gpu" || arg == "--gpu-contracted" || arg == "--compare") {
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
            const auto prepared = gsc::prepare_case_cpu<std::uint32_t>(cfg);
            const auto cpu_result = gsc::solve_backward_cpu(prepared, cfg.max_iterations, cfg.verbose);
            print_summary(prepared, cpu_result, "cpu");
            gsc::write_results_hdf5(output_path, cfg, prepared, {{"cpu", &cpu_result}});
            std::cout << "result file   : " << output_path << '\n';
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
            
            // 使用GPU专用的HDF5写入函数，包含抽象数据和逐迭代统计
            gsc::write_gpu_results_hdf5(output_path, cfg, gpu);
            std::cout << "result file   : " << output_path << '\n';
            std::cout << "  - includes abstraction data (" 
                      << (gpu.pair_count * (2 * sizeof(std::uint32_t) + sizeof(std::uint8_t)) / (1024.0 * 1024.0))
                      << " MB)\n";
            std::cout << "  - includes " << gpu.iteration_stats.size() << " iteration statistics\n";
            return 0;
        }

        if (mode == "--gpu-contracted") {
            // 收缩约束前向可达性模式
            const auto gpu = gsc::run_contracted_case_cuda_u32(cfg);
            if (!gpu.executed) {
                std::cerr << "gpu contracted run unavailable: " << gpu.message << '\n';
                return 2;
            }
            print_summary(gpu.state_grid, gpu.input_grid, gpu.budget, gpu.pair_count,
                         gpu.result.candidate_mask, gpu.result, gpu.abstraction_ms, "gpu-contracted");
            
            gsc::write_gpu_results_hdf5(output_path, cfg, gpu);
            std::cout << "result file   : " << output_path << '\n';
            std::cout << "  - includes abstraction data (" 
                      << (gpu.pair_count * (2 * sizeof(std::uint32_t) + sizeof(std::uint8_t)) / (1024.0 * 1024.0))
                      << " MB)\n";
            std::cout << "  - includes " << gpu.iteration_stats.size() << " iteration statistics\n";
            return 0;
        }

        if (mode == "--compare") {
            const auto prepared = gsc::prepare_case_cpu<std::uint32_t>(cfg);
            const auto cpu_result = gsc::solve_backward_cpu(prepared, cfg.max_iterations, cfg.verbose);
            const auto gpu = gsc::run_case_cuda_u32(cfg);
            if (!gpu.executed) {
                std::cerr << "gpu compare unavailable: " << gpu.message << '\n';
                print_summary(prepared, cpu_result, "cpu");
                gsc::write_results_hdf5(output_path, cfg, prepared, {{"cpu", &cpu_result}});
                std::cout << "result file   : " << output_path << '\n';
                return 2;
            }

            print_summary(prepared, cpu_result, "cpu");
            print_summary(prepared, gpu.result, "gpu");
            gsc::write_results_hdf5(output_path, cfg, prepared, {{"cpu", &cpu_result}, {"gpu", &gpu.result}});
            std::cout << "result file   : " << output_path << '\n';
            std::cout << "exact_match   : " << (same_result(cpu_result, gpu.result) ? "true" : "false") << '\n';
            return same_result(cpu_result, gpu.result) ? 0 : 3;
        }

        throw std::runtime_error("unknown mode: " + mode);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
