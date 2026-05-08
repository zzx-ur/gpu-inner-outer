#include <chrono>
#include <iostream>
#include <stdexcept>

#include "gsc/cuda_api.hpp"

int main() {
    try {
        const auto cfg = gsc::make_smoke_case();
        const auto prepared = gsc::prepare_case_cpu<std::uint32_t>(cfg);
        if (prepared.abstraction.pair_count <= 1000000ULL) {
            throw std::runtime_error("smoke test case did not exceed 1e6 pairs");
        }

        const auto t0 = std::chrono::steady_clock::now();
        const auto result = gsc::solve_backward_cpu(prepared, cfg.max_iterations, false);
        const auto t1 = std::chrono::steady_clock::now();

        std::cout << "smoke pairs     : " << prepared.abstraction.pair_count << '\n';
        std::cout << "memory estimate : " << gsc::format_bytes(prepared.budget.total_bytes) << '\n';
        std::cout << "cpu total time  : "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count()
                  << " ms\n";
        std::cout << "reachable states: " << result.reachable_states << '\n';

        if (gsc::cuda_backend_compiled()) {
            const auto gpu = gsc::run_case_cuda_u32(cfg);
            if (gpu.executed) {
                std::cout << "gpu abstraction : " << gpu.abstraction_ms << " ms\n";
                std::cout << "gpu solve       : " << gpu.solve_ms << " ms\n";
            } else {
                std::cout << "gpu smoke skipped: " << gpu.message << '\n';
            }
        } else {
            std::cout << "gpu smoke skipped: cuda backend not built\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "smoke_test failed: " << ex.what() << '\n';
        return 1;
    }
}
