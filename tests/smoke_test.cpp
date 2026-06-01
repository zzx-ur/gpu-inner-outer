#include <chrono>
#include <iostream>
#include <stdexcept>

#include "gsc/cuda_api.hpp"

int main() {
    try {
        const auto cfg = gsc::make_smoke_case();
        
        // GPU-only computation
        const auto gpu = gsc::run_case_cuda_u32(cfg);
        if (!gpu.executed) {
            throw std::runtime_error("GPU smoke test failed: " + gpu.message);
        }

        if (gpu.pair_count <= 1000000ULL) {
            throw std::runtime_error("smoke test case did not exceed 1e6 pairs");
        }

        std::cout << "smoke pairs     : " << gpu.pair_count << '\n';
        std::cout << "memory estimate : " << gsc::format_bytes(gpu.budget.total_bytes) << '\n';
        std::cout << "gpu abstraction : " << gpu.abstraction_ms << " ms\n";
        std::cout << "gpu solve       : " << gpu.solve_ms << " ms\n";
        std::cout << "gpu total       : " << gpu.total_ms << " ms\n";
        std::cout << "reachable states: " << gpu.result.reachable_states << '\n';

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "smoke_test failed: " << ex.what() << '\n';
        return 1;
    }
}
