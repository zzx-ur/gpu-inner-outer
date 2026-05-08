#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "gsc/abstraction.hpp"
#include "gsc/prefix_sum.hpp"

namespace gsc {

template <typename StateIndex>
struct SolveResult {
    std::vector<std::uint8_t> valid_mask;
    std::vector<std::uint8_t> candidate_mask;
    std::vector<std::uint8_t> reachable_mask;
    std::vector<InputIndex> controller;
    std::vector<ReachStep> reach_step;
    std::uint32_t iterations = 0;
    bool converged = false;
    std::string message;
    std::uint64_t reachable_states = 0;
    std::uint64_t certified_candidate_states = 0;
    double solve_ms = 0.0;
};

template <typename StateIndex>
inline SolveResult<StateIndex> solve_backward_cpu(const PreparedCase<StateIndex>& prepared,
                                                  std::uint32_t max_iterations,
                                                  bool verbose) {
    SolveResult<StateIndex> result;
    result.valid_mask = prepared.valid_mask;
    result.candidate_mask = prepared.candidate_mask;
    result.reachable_mask = prepared.candidate_mask;
    result.controller.assign(prepared.state_grid.total_size, kInvalidInput);
    result.reach_step.assign(prepared.state_grid.total_size, kUnreachableStep);

    std::uint64_t candidate_count = 0;
    for (std::uint64_t i = 0; i < prepared.state_grid.total_size; ++i) {
        if (prepared.candidate_mask[i]) {
            result.reach_step[i] = 0;
            ++candidate_count;
        }
    }

    const auto prefix_valid = build_prefix_sum_cpu(prepared.state_grid, prepared.valid_mask);
    std::uint64_t certified_candidate_states = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (std::uint32_t iter = 1; iter <= max_iterations; ++iter) {
        const auto prefix_reachable = build_prefix_sum_cpu(prepared.state_grid, result.reachable_mask);
        auto reachable_next = result.reachable_mask;
        std::uint64_t new_reachable = 0;
        std::uint64_t new_candidate_certified = 0;

        for (std::uint64_t state = 0; state < prepared.state_grid.total_size; ++state) {
            if (!prepared.valid_mask[state]) {
                continue;
            }

            const bool is_candidate = prepared.candidate_mask[state] != 0;
            const bool already_reached = result.reachable_mask[state] != 0;
            const bool already_certified = result.controller[state] != kInvalidInput;
            if ((!is_candidate && already_reached) || (is_candidate && already_certified)) {
                continue;
            }

            const std::uint64_t pair_base = state * prepared.input_grid.total_size;
            for (std::uint64_t input = 0; input < prepared.input_grid.total_size; ++input) {
                const std::uint64_t pair = pair_base + input;
                if (!prepared.abstraction.valid[pair]) {
                    continue;
                }

                const auto valid_count =
                    query_box_count(prepared.state_grid,
                                    prefix_valid,
                                    prepared.abstraction.min_flat[pair],
                                    prepared.abstraction.max_flat[pair]);
                if (valid_count == 0) {
                    continue;
                }

                const auto reachable_count =
                    query_box_count(prepared.state_grid,
                                    prefix_reachable,
                                    prepared.abstraction.min_flat[pair],
                                    prepared.abstraction.max_flat[pair]);

                if (reachable_count == valid_count) {
                    if (!already_reached) {
                        reachable_next[state] = 1;
                        ++new_reachable;
                    }
                    if (!already_certified) {
                        result.controller[state] = static_cast<InputIndex>(input);
                        if (is_candidate) {
                            ++new_candidate_certified;
                        } else {
                            result.reach_step[state] = iter;
                        }
                    }
                    break;
                }
            }
        }

        result.reachable_mask.swap(reachable_next);
        certified_candidate_states += new_candidate_certified;
        result.iterations = iter;

        if (verbose) {
            std::uint64_t candidate_reached = 0;
            for (std::uint64_t i = 0; i < prepared.state_grid.total_size; ++i) {
                if (prepared.candidate_mask[i] && result.controller[i] != kInvalidInput) {
                    ++candidate_reached;
                }
            }
            const double coverage = candidate_count == 0
                                        ? 100.0
                                        : 100.0 * static_cast<double>(candidate_reached) / static_cast<double>(candidate_count);
            std::printf("iter %u: new_reachable=%llu candidate_controller_coverage=%.1f%% (%llu/%llu)\n",
                        iter,
                        static_cast<unsigned long long>(new_reachable),
                        coverage,
                        static_cast<unsigned long long>(candidate_reached),
                        static_cast<unsigned long long>(candidate_count));
        }

        if (certified_candidate_states == candidate_count) {
            result.converged = true;
            result.message = "all candidate states obtained a controller";
            break;
        }

        if (new_reachable == 0 && new_candidate_certified == 0) {
            result.message = "iteration stalled before candidate closure";
            break;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    result.solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.reachable_states = std::count(result.reachable_mask.begin(), result.reachable_mask.end(), std::uint8_t{1});
    result.certified_candidate_states = certified_candidate_states;

    if (!result.converged && result.message.empty()) {
        result.message = "maximum iterations reached";
    }

    return result;
}

}  // namespace gsc
