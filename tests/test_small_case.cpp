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

        const auto prepared = gsc::prepare_case_cpu<std::uint32_t>(base_cfg);
        const auto result = gsc::solve_backward_cpu(prepared, base_cfg.max_iterations, false);

        expect(!prepared.valid_mask.empty(), "valid mask should not be empty");
        expect(!prepared.candidate_mask.empty(), "candidate mask should not be empty");
        expect(prepared.abstraction.pair_count == prepared.state_grid.total_size * prepared.input_grid.total_size,
               "pair count mismatch");

        std::uint64_t candidate_states = 0;
        for (auto v : prepared.candidate_mask) {
            candidate_states += (v != 0);
        }
        expect(candidate_states > 0, "candidate set must not be empty");

        const auto prepared_wrap = gsc::prepare_case_cpu<std::uint32_t>(wrap_cfg);
        const auto wrap_result = gsc::solve_backward_cpu(prepared_wrap, wrap_cfg.max_iterations, false);
        expect(!wrap_result.reachable_mask.empty(), "wraparound result must not be empty");

        if (gsc::cuda_backend_compiled()) {
            const auto gpu = gsc::run_case_cuda_u32(base_cfg);
            expect(gpu.executed, "gpu backend compiled but run did not execute");
            expect(same_result(result, gpu.result), "cpu/gpu mismatch on small case");

            const auto gpu_wrap = gsc::run_case_cuda_u32(wrap_cfg);
            expect(gpu_wrap.executed, "gpu wraparound run did not execute");
            expect(same_result(wrap_result, gpu_wrap.result), "cpu/gpu mismatch on wraparound case");
        } else {
            std::cout << "cuda backend not built; cpu-only checks passed\n";
        }

        std::cout << "small-case correctness checks passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "test_small_case failed: " << ex.what() << '\n';
        return 1;
    }
}
