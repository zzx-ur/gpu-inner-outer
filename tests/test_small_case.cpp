#include <iostream>
#include <stdexcept>

#include "gsc/cuda_api.hpp"

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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

int main() {
    try {
        const auto base_cfg = gsc::make_default_case();
        const auto wrap_cfg = gsc::make_wraparound_case();

        // GPU-only computation for base case
        const auto gpu = gsc::run_case_cuda_u32(base_cfg);
        expect(gpu.executed, "gpu backend run did not execute");
        
        expect(!gpu.result.valid_mask.empty(), "valid mask should not be empty");
        expect(!gpu.result.candidate_mask.empty(), "candidate mask should not be empty");
        expect(gpu.pair_count == gpu.state_grid.total_size * gpu.input_grid.total_size,
               "pair count mismatch");

        std::uint64_t candidate_states = 0;
        for (auto v : gpu.result.candidate_mask) {
            candidate_states += (v != 0);
        }
        expect(candidate_states > 0, "candidate set must not be empty");

        // GPU-only computation for wraparound case
        const auto gpu_wrap = gsc::run_case_cuda_u32(wrap_cfg);
        expect(gpu_wrap.executed, "gpu wraparound run did not execute");
        expect(!gpu_wrap.result.reachable_mask.empty(), "wraparound result must not be empty");

        std::cout << "small-case correctness checks passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "test_small_case failed: " << ex.what() << '\n';
        return 1;
    }
}
