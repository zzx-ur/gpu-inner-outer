#include "gsc/cuda_api.hpp"

#if GSC_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gsc/abstraction.hpp"
#include "gsc/prefix_sum.hpp"

namespace gsc {

namespace {

#define GSC_CUDA_CHECK(expr)                                                      \
    do {                                                                          \
        cudaError_t err__ = (expr);                                               \
        if (err__ != cudaSuccess) {                                               \
            throw std::runtime_error(cudaGetErrorString(err__));                  \
        }                                                                         \
    } while (0)

// 显存监控辅助函数
inline std::uint64_t get_gpu_memory_used() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    GSC_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return total_bytes - free_bytes;
}

inline void record_memory_stats(GpuMemoryStats& stats, const std::string& phase) {
    std::uint64_t used = get_gpu_memory_used();
    if (used > stats.peak_usage_bytes) {
        stats.peak_usage_bytes = used;
    }
}

// GPU 端约束参数结构（POD 类型，可直接传递给 kernel）
struct GpuConstraintParams {
    int constraint_type;  // 0=none, 1=hyperbolic, 2=elliptic
    double hyperbolic_a;
    double hyperbolic_b;
    double hyperbolic_c;
    double elliptic_a;
    double elliptic_b;
};

// 设备端约束检查函数
__device__ bool d_satisfies_constraint(const double x[kStateDim], const GpuConstraintParams& params) {
    switch (params.constraint_type) {
        case 0:  // kNone
            return true;
        case 1: {  // kHyperbolic: x1^2 - x2^2 <= a 且 b*x2^2 - x1^2 <= c
            const double x1_sq = x[0] * x[0];
            const double x2_sq = x[1] * x[1];
            return (x1_sq - x2_sq <= params.hyperbolic_a) && 
                   (params.hyperbolic_b * x2_sq - x1_sq <= params.hyperbolic_c);
        }
        case 2: {  // kElliptic: (x1/a)^2 + (x2/b)^2 <= 1
            const double term1 = (x[0] / params.elliptic_a) * (x[0] / params.elliptic_a);
            const double term2 = (x[1] / params.elliptic_b) * (x[1] / params.elliptic_b);
            return (term1 + term2 <= 1.0);
        }
        default:
            return true;
    }
}

// 前向声明
__device__ PrefixCount d_query_box_count(const Grid4D<std::uint32_t>& grid,
                                         const PrefixLayout4D& layout,
                                         const PrefixCount* data,
                                         std::uint32_t min_flat,
                                         std::uint32_t max_flat);

// 使用前缀和快速检查盒内所有状态是否满足约束
// 返回 true 表示盒内所有有效状态都满足约束
__device__ bool d_box_satisfies_constraint_fast(const Grid4D<std::uint32_t>& grid,
                                                const PrefixLayout4D& layout,
                                                const PrefixCount* prefix_constraint,
                                                const PrefixCount* prefix_valid,
                                                const IndexBox4D& box) {
    // 查询盒内满足约束的状态数
    const PrefixCount constraint_count = d_query_box_count(grid, layout, prefix_constraint, 
                                                           grid.flatten(box.min), grid.flatten(box.max));
    
    // 查询盒内有效的状态数
    const PrefixCount valid_count = d_query_box_count(grid, layout, prefix_valid,
                                                      grid.flatten(box.min), grid.flatten(box.max));
    
    // 所有有效状态都满足约束当且仅当两个计数相等
    return constraint_count == valid_count;
}

__device__ PrefixCount d_query_box_single(const PrefixLayout4D& layout,
                                          const PrefixCount* data,
                                          const IndexBox4D& box) {
    const std::uint32_t l[4] = {box.min[0], box.min[1], box.min[2], box.min[3]};
    const std::uint32_t r[4] = {box.max[0] + 1, box.max[1] + 1, box.max[2] + 1, box.max[3] + 1};
    PrefixCount total = 0;
    for (int mask = 0; mask < 16; ++mask) {
        const std::uint32_t x0 = (mask & 1) ? l[0] : r[0];
        const std::uint32_t x1 = (mask & 2) ? l[1] : r[1];
        const std::uint32_t x2 = (mask & 4) ? l[2] : r[2];
        const std::uint32_t x3 = (mask & 8) ? l[3] : r[3];
        const PrefixCount value = data[layout.offset(x0, x1, x2, x3)];
        if ((__popc(mask) & 1) == 0) {
            total += value;
        } else {
            total -= value;
        }
    }
    return total;
}

__device__ PrefixCount d_query_box_count(const Grid4D<std::uint32_t>& grid,
                                         const PrefixLayout4D& layout,
                                         const PrefixCount* data,
                                         std::uint32_t min_flat,
                                         std::uint32_t max_flat) {
    IndexBox4D box{};
    index_box_from_flats(grid, min_flat, max_flat, &box);
    if (box.min[grid.wrap_dim] <= box.max[grid.wrap_dim]) {
        return d_query_box_single(layout, data, box);
    }
    IndexBox4D left = box;
    IndexBox4D right = box;
    left.max[grid.wrap_dim] = grid.shape[grid.wrap_dim] - 1;
    right.min[grid.wrap_dim] = 0;
    return d_query_box_single(layout, data, left) + d_query_box_single(layout, data, right);
}

__global__ void abstraction_kernel(Grid4D<std::uint32_t> state_grid,
                                   InputGrid2D input_grid,
                                   Unicycle4DModel model,
                                   GpuConstraintParams constraint_params,
                                   PrefixLayout4D layout,
                                   const PrefixCount* prefix_constraint,
                                   const PrefixCount* prefix_valid,
                                   std::uint32_t* min_flat,
                                   std::uint32_t* max_flat,
                                   std::uint8_t* valid,
                                   std::uint64_t pair_count) {
    const std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= pair_count) {
        return;
    }

    const std::uint64_t state = tid / input_grid.total_size;
    const std::uint64_t input = tid % input_grid.total_size;

    double x[kStateDim];
    double u[kInputDim];
    double min_coord[kStateDim];
    double max_coord[kStateDim];

    state_grid.center(static_cast<std::uint32_t>(state), x);
    
    // 检查当前状态是否满足约束
    if (!d_satisfies_constraint(x, constraint_params)) {
        valid[tid] = 0;
        min_flat[tid] = 0;
        max_flat[tid] = 0;
        return;
    }
    
    input_grid.center(static_cast<InputIndex>(input), u);
    model.successor_box(x, u, state_grid.eta, min_coord, max_coord);

    IndexBox4D box{};
    if (!continuous_box_to_index_box(state_grid, min_coord, max_coord, &box)) {
        valid[tid] = 0;
        min_flat[tid] = 0;
        max_flat[tid] = 0;
        return;
    }

    // 使用前缀和快速检查后继盒内所有状态是否满足约束
    if (!d_box_satisfies_constraint_fast(state_grid, layout, prefix_constraint, prefix_valid, box)) {
        valid[tid] = 0;
        min_flat[tid] = 0;
        max_flat[tid] = 0;
        return;
    }

    valid[tid] = 1;
    min_flat[tid] = state_grid.flatten(box.min);
    max_flat[tid] = state_grid.flatten(box.max);
}

__global__ void build_constraint_mask_kernel(const Grid4D<std::uint32_t> state_grid,
                                              const GpuConstraintParams constraint_params,
                                              std::uint8_t* constraint_mask) {
    const std::uint64_t flat = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= state_grid.total_size) {
        return;
    }

    double center[kStateDim];
    state_grid.center(static_cast<std::uint32_t>(flat), center);
    constraint_mask[flat] = d_satisfies_constraint(center, constraint_params) ? 1 : 0;
}

__global__ void scatter_mask_kernel(const Grid4D<std::uint32_t> state_grid,
                                    const PrefixLayout4D layout,
                                    const std::uint8_t* mask,
                                    PrefixCount* prefix) {
    const std::uint64_t flat = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= state_grid.total_size) {
        return;
    }

    std::uint32_t idx[4];
    state_grid.unflatten(static_cast<std::uint32_t>(flat), idx);
    const auto offset = layout.offset(idx[0] + 1, idx[1] + 1, idx[2] + 1, idx[3] + 1);
    prefix[offset] = mask[flat] ? 1 : 0;
}

__global__ void scan_axis0_kernel(PrefixCount* data, PrefixLayout4D layout) {
    const std::uint64_t line = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t line_count =
        static_cast<std::uint64_t>(layout.padded_shape[1]) *
        layout.padded_shape[2] *
        layout.padded_shape[3];
    if (line >= line_count) {
        return;
    }

    const std::uint32_t y = static_cast<std::uint32_t>(line % layout.padded_shape[1]);
    const std::uint32_t z = static_cast<std::uint32_t>((line / layout.padded_shape[1]) % layout.padded_shape[2]);
    const std::uint32_t w = static_cast<std::uint32_t>(line / (layout.padded_shape[1] * layout.padded_shape[2]));
    const auto base = layout.offset(0, y, z, w);
    for (std::uint32_t x = 1; x < layout.padded_shape[0]; ++x) {
        data[base + x] += data[base + x - 1];
    }
}

__global__ void scan_axis1_kernel(PrefixCount* data, PrefixLayout4D layout) {
    const std::uint64_t line = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t line_count =
        static_cast<std::uint64_t>(layout.padded_shape[0]) *
        layout.padded_shape[2] *
        layout.padded_shape[3];
    if (line >= line_count) {
        return;
    }
    const std::uint32_t x = static_cast<std::uint32_t>(line % layout.padded_shape[0]);
    const std::uint32_t z = static_cast<std::uint32_t>((line / layout.padded_shape[0]) % layout.padded_shape[2]);
    const std::uint32_t w = static_cast<std::uint32_t>(line / (layout.padded_shape[0] * layout.padded_shape[2]));
    const auto base = layout.offset(x, 0, z, w);
    for (std::uint32_t y = 1; y < layout.padded_shape[1]; ++y) {
        data[base + static_cast<std::uint64_t>(y) * layout.strides[1]] +=
            data[base + static_cast<std::uint64_t>(y - 1) * layout.strides[1]];
    }
}

__global__ void scan_axis2_kernel(PrefixCount* data, PrefixLayout4D layout) {
    const std::uint64_t line = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t line_count =
        static_cast<std::uint64_t>(layout.padded_shape[0]) *
        layout.padded_shape[1] *
        layout.padded_shape[3];
    if (line >= line_count) {
        return;
    }
    const std::uint32_t x = static_cast<std::uint32_t>(line % layout.padded_shape[0]);
    const std::uint32_t y = static_cast<std::uint32_t>((line / layout.padded_shape[0]) % layout.padded_shape[1]);
    const std::uint32_t w = static_cast<std::uint32_t>(line / (layout.padded_shape[0] * layout.padded_shape[1]));
    const auto base = layout.offset(x, y, 0, w);
    for (std::uint32_t z = 1; z < layout.padded_shape[2]; ++z) {
        data[base + static_cast<std::uint64_t>(z) * layout.strides[2]] +=
            data[base + static_cast<std::uint64_t>(z - 1) * layout.strides[2]];
    }
}

__global__ void scan_axis3_kernel(PrefixCount* data, PrefixLayout4D layout) {
    const std::uint64_t line = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t line_count =
        static_cast<std::uint64_t>(layout.padded_shape[0]) *
        layout.padded_shape[1] *
        layout.padded_shape[2];
    if (line >= line_count) {
        return;
    }
    const std::uint32_t x = static_cast<std::uint32_t>(line % layout.padded_shape[0]);
    const std::uint32_t y = static_cast<std::uint32_t>((line / layout.padded_shape[0]) % layout.padded_shape[1]);
    const std::uint32_t z = static_cast<std::uint32_t>(line / (layout.padded_shape[0] * layout.padded_shape[1]));
    const auto base = layout.offset(x, y, z, 0);
    for (std::uint32_t w = 1; w < layout.padded_shape[3]; ++w) {
        data[base + static_cast<std::uint64_t>(w) * layout.strides[3]] +=
            data[base + static_cast<std::uint64_t>(w - 1) * layout.strides[3]];
    }
}

__global__ void pair_satisfaction_kernel(Grid4D<std::uint32_t> state_grid,
                                         PrefixLayout4D layout,
                                         const PrefixCount* prefix_reachable,
                                         const PrefixCount* prefix_valid,
                                         const std::uint32_t* min_flat,
                                         const std::uint32_t* max_flat,
                                         const std::uint8_t* pair_valid,
                                         std::uint8_t* pair_satisfied,
                                         std::uint64_t pair_count) {
    const std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= pair_count) {
        return;
    }
    if (!pair_valid[tid]) {
        pair_satisfied[tid] = 0;
        return;
    }
    const PrefixCount valid_count =
        d_query_box_count(state_grid, layout, prefix_valid, min_flat[tid], max_flat[tid]);
    if (valid_count == 0) {
        pair_satisfied[tid] = 0;
        return;
    }
    const PrefixCount reachable_count =
        d_query_box_count(state_grid, layout, prefix_reachable, min_flat[tid], max_flat[tid]);
    pair_satisfied[tid] = (reachable_count == valid_count) ? 1 : 0;
}

__global__ void reduce_inputs_kernel(const std::uint8_t* valid_state_mask,
                                     const std::uint8_t* candidate_mask,
                                     const std::uint8_t* reachable_prev,
                                     std::uint8_t* reachable_next,
                                     InputIndex* controller,
                                     ReachStep* reach_step,
                                     const std::uint8_t* pair_satisfied,
                                     std::uint64_t state_count,
                                     std::uint32_t input_count,
                                     std::uint32_t iter,
                                     unsigned long long* new_reachable,
                                     unsigned long long* new_candidate_certified) {
    const std::uint64_t state = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (state >= state_count) {
        return;
    }

    reachable_next[state] = reachable_prev[state];
    if (!valid_state_mask[state]) {
        return;
    }

    const bool is_candidate = candidate_mask[state] != 0;
    const bool already_reached = reachable_prev[state] != 0;
    const bool already_certified = controller[state] != kInvalidInput;
    if ((!is_candidate && already_reached) || (is_candidate && already_certified)) {
        return;
    }

    const std::uint64_t pair_base = state * input_count;
    InputIndex chosen = kInvalidInput;
    for (std::uint32_t input = 0; input < input_count; ++input) {
        if (pair_satisfied[pair_base + input]) {
            chosen = input;
            break;
        }
    }

    if (chosen == kInvalidInput) {
        return;
    }

    if (!already_reached) {
        reachable_next[state] = 1;
        atomicAdd(new_reachable, 1ULL);
    }
    if (!already_certified) {
        controller[state] = chosen;
        if (is_candidate) {
            atomicAdd(new_candidate_certified, 1ULL);
            // reach_step[state] already initialized to 0 for candidate states
        } else if (reach_step[state] == kUnreachableStep) {
            reach_step[state] = iter;
        }
    }
}

std::uint32_t ceil_div_to_u32(std::uint64_t value, std::uint32_t divisor) {
    const std::uint64_t blocks = (value + divisor - 1) / divisor;
    if (blocks > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("kernel launch grid exceeds uint32_t");
    }
    return static_cast<std::uint32_t>(blocks);
}

void build_prefix_on_device(const Grid4D<std::uint32_t>& grid,
                             PrefixLayout4D layout,
                             const std::uint8_t* mask,
                             PrefixCount* prefix,
                             KernelTimings& kernel_timings) {
    GSC_CUDA_CHECK(cudaMemset(prefix, 0, layout.total_size * sizeof(PrefixCount)));

    const int threads = 256;
    cudaEvent_t event_start, event_stop;
    GSC_CUDA_CHECK(cudaEventCreate(&event_start));
    GSC_CUDA_CHECK(cudaEventCreate(&event_stop));

    const auto scatter_blocks = ceil_div_to_u32(grid.total_size, threads);
    
    // Time scatter_mask_kernel
    GSC_CUDA_CHECK(cudaEventRecord(event_start));
    scatter_mask_kernel<<<scatter_blocks, threads>>>(grid, layout, mask, prefix);
     GSC_CUDA_CHECK(cudaGetLastError());
     GSC_CUDA_CHECK(cudaEventRecord(event_stop));
     GSC_CUDA_CHECK(cudaEventSynchronize(event_stop));
     float scatter_time = 0.0f;
     GSC_CUDA_CHECK(cudaEventElapsedTime(&scatter_time, event_start, event_stop));
     kernel_timings.scatter_mask_ms += static_cast<double>(scatter_time);

    const std::uint64_t lines0 = static_cast<std::uint64_t>(layout.padded_shape[1]) *
                                 layout.padded_shape[2] *
                                 layout.padded_shape[3];
    const std::uint64_t lines1 = static_cast<std::uint64_t>(layout.padded_shape[0]) *
                                 layout.padded_shape[2] *
                                 layout.padded_shape[3];
    const std::uint64_t lines2 = static_cast<std::uint64_t>(layout.padded_shape[0]) *
                                 layout.padded_shape[1] *
                                 layout.padded_shape[3];
    const std::uint64_t lines3 = static_cast<std::uint64_t>(layout.padded_shape[0]) *
                                 layout.padded_shape[1] *
                                 layout.padded_shape[2];

    // Time scan_axis0_kernel
    GSC_CUDA_CHECK(cudaEventRecord(event_start));
    scan_axis0_kernel<<<ceil_div_to_u32(lines0, threads), threads>>>(prefix, layout);
    GSC_CUDA_CHECK(cudaGetLastError());
    GSC_CUDA_CHECK(cudaEventRecord(event_stop));
    GSC_CUDA_CHECK(cudaEventSynchronize(event_stop));
    float scan0_time = 0.0f;
    GSC_CUDA_CHECK(cudaEventElapsedTime(&scan0_time, event_start, event_stop));
    kernel_timings.scan_axis0_ms += static_cast<double>(scan0_time);

    // Time scan_axis1_kernel
    GSC_CUDA_CHECK(cudaEventRecord(event_start));
    scan_axis1_kernel<<<ceil_div_to_u32(lines1, threads), threads>>>(prefix, layout);
    GSC_CUDA_CHECK(cudaGetLastError());
    GSC_CUDA_CHECK(cudaEventRecord(event_stop));
    GSC_CUDA_CHECK(cudaEventSynchronize(event_stop));
    float scan1_time = 0.0f;
    GSC_CUDA_CHECK(cudaEventElapsedTime(&scan1_time, event_start, event_stop));
    kernel_timings.scan_axis1_ms += static_cast<double>(scan1_time);

    // Time scan_axis2_kernel
    GSC_CUDA_CHECK(cudaEventRecord(event_start));
    scan_axis2_kernel<<<ceil_div_to_u32(lines2, threads), threads>>>(prefix, layout);
    GSC_CUDA_CHECK(cudaGetLastError());
    GSC_CUDA_CHECK(cudaEventRecord(event_stop));
    GSC_CUDA_CHECK(cudaEventSynchronize(event_stop));
    float scan2_time = 0.0f;
    GSC_CUDA_CHECK(cudaEventElapsedTime(&scan2_time, event_start, event_stop));
    kernel_timings.scan_axis2_ms += static_cast<double>(scan2_time);

    // Time scan_axis3_kernel
    GSC_CUDA_CHECK(cudaEventRecord(event_start));
    scan_axis3_kernel<<<ceil_div_to_u32(lines3, threads), threads>>>(prefix, layout);
    GSC_CUDA_CHECK(cudaGetLastError());
    GSC_CUDA_CHECK(cudaEventRecord(event_stop));
    GSC_CUDA_CHECK(cudaEventSynchronize(event_stop));
    float scan3_time = 0.0f;
    GSC_CUDA_CHECK(cudaEventElapsedTime(&scan3_time, event_start, event_stop));
    kernel_timings.scan_axis3_ms += static_cast<double>(scan3_time);

    GSC_CUDA_CHECK(cudaGetLastError());
    
    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
}

}  // namespace

bool cuda_backend_compiled() {
    return true;
}

GpuRunReport run_case_cuda_u32(const CaseConfig& cfg) {
    GpuRunReport report;
    report.compiled_with_cuda = true;

    int device_count = 0;
    GSC_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        report.message = "no CUDA device found";
        return report;
    }
    report.runtime_available = true;

    // GPU-only 模式：仅构建网格和 mask，跳过 CPU 抽象
    const auto prepared = prepare_case_gpu_minimal<std::uint32_t>(cfg);
    
    // 保存网格信息到 report（用于输出）
    report.state_grid = prepared.state_grid;
    report.input_grid = prepared.input_grid;
    report.budget = prepared.budget;
    report.pair_count = prepared.state_grid.total_size * prepared.input_grid.total_size;

    std::uint32_t* d_min_flat = nullptr;
    std::uint32_t* d_max_flat = nullptr;
    std::uint8_t* d_pair_valid = nullptr;
    std::uint8_t* d_pair_satisfied = nullptr;
    std::uint8_t* d_valid_mask = nullptr;
    std::uint8_t* d_candidate_mask = nullptr;
    std::uint8_t* d_constraint_mask = nullptr;
    std::uint8_t* d_reachable_prev = nullptr;
    std::uint8_t* d_reachable_next = nullptr;
    InputIndex* d_controller = nullptr;
    ReachStep* d_reach_step = nullptr;
    PrefixCount* d_prefix_valid = nullptr;
    PrefixCount* d_prefix_constraint = nullptr;
    PrefixCount* d_prefix_reachable = nullptr;
    unsigned long long* d_new_reachable = nullptr;
    unsigned long long* d_new_candidate_certified = nullptr;

    try {
        const auto pair_count = prepared.abstraction.pair_count;
        const auto state_count = prepared.state_grid.total_size;
        const auto prefix_layout = build_prefix_layout(prepared.state_grid);
        const int threads = 256;

        // 计算显存分配量
        std::uint64_t abstraction_mem = pair_count * sizeof(std::uint32_t) * 2 + pair_count * sizeof(std::uint8_t);
        std::uint64_t prefix_mem = prefix_layout.total_size * sizeof(PrefixCount) * 3;  // valid, constraint, reachable
        std::uint64_t iteration_mem = state_count * sizeof(std::uint8_t) * 5 +  // valid, candidate, constraint, reachable_prev, reachable_next
                                      state_count * sizeof(InputIndex) + 
                                      state_count * sizeof(ReachStep) + 
                                      sizeof(unsigned long long) * 2;
        
        report.memory_stats.abstraction_bytes = abstraction_mem;
        report.memory_stats.prefix_bytes = prefix_mem;
        report.memory_stats.iteration_bytes = iteration_mem;
        report.memory_stats.total_allocated_bytes = abstraction_mem + prefix_mem + iteration_mem;

        GSC_CUDA_CHECK(cudaMalloc(&d_min_flat, pair_count * sizeof(std::uint32_t)));
        GSC_CUDA_CHECK(cudaMalloc(&d_max_flat, pair_count * sizeof(std::uint32_t)));
        GSC_CUDA_CHECK(cudaMalloc(&d_pair_valid, pair_count * sizeof(std::uint8_t)));
        GSC_CUDA_CHECK(cudaMalloc(&d_pair_satisfied, pair_count * sizeof(std::uint8_t)));
        GSC_CUDA_CHECK(cudaMalloc(&d_valid_mask, state_count * sizeof(std::uint8_t)));
        GSC_CUDA_CHECK(cudaMalloc(&d_candidate_mask, state_count * sizeof(std::uint8_t)));
        GSC_CUDA_CHECK(cudaMalloc(&d_constraint_mask, state_count * sizeof(std::uint8_t)));
        GSC_CUDA_CHECK(cudaMalloc(&d_reachable_prev, state_count * sizeof(std::uint8_t)));
        GSC_CUDA_CHECK(cudaMalloc(&d_reachable_next, state_count * sizeof(std::uint8_t)));
        GSC_CUDA_CHECK(cudaMalloc(&d_controller, state_count * sizeof(InputIndex)));
        GSC_CUDA_CHECK(cudaMalloc(&d_reach_step, state_count * sizeof(ReachStep)));
        GSC_CUDA_CHECK(cudaMalloc(&d_prefix_valid, prefix_layout.total_size * sizeof(PrefixCount)));
        GSC_CUDA_CHECK(cudaMalloc(&d_prefix_constraint, prefix_layout.total_size * sizeof(PrefixCount)));
        GSC_CUDA_CHECK(cudaMalloc(&d_prefix_reachable, prefix_layout.total_size * sizeof(PrefixCount)));
        GSC_CUDA_CHECK(cudaMalloc(&d_new_reachable, sizeof(unsigned long long)));
        GSC_CUDA_CHECK(cudaMalloc(&d_new_candidate_certified, sizeof(unsigned long long)));

        record_memory_stats(report.memory_stats, "after allocation");

        // 测量初始化阶段的memcpy
        auto init_memcpy_start = std::chrono::steady_clock::now();
        
        GSC_CUDA_CHECK(cudaMemcpy(d_valid_mask,
                                  prepared.valid_mask.data(),
                                  state_count * sizeof(std::uint8_t),
                                  cudaMemcpyHostToDevice));
        GSC_CUDA_CHECK(cudaMemcpy(d_candidate_mask,
                                  prepared.candidate_mask.data(),
                                  state_count * sizeof(std::uint8_t),
                                  cudaMemcpyHostToDevice));

        std::vector<std::uint8_t> reachable_seed = prepared.candidate_mask;
        std::vector<InputIndex> controller_seed(state_count, kInvalidInput);
        std::vector<ReachStep> reach_step_seed(state_count, kUnreachableStep);
        std::uint64_t candidate_count = 0;
        for (std::uint64_t i = 0; i < state_count; ++i) {
            if (prepared.candidate_mask[i]) {
                reach_step_seed[i] = 0;
                ++candidate_count;
            }
        }

        GSC_CUDA_CHECK(cudaMemcpy(d_reachable_prev,
                                  reachable_seed.data(),
                                  state_count * sizeof(std::uint8_t),
                                  cudaMemcpyHostToDevice));
        GSC_CUDA_CHECK(cudaMemcpy(d_controller,
                                  controller_seed.data(),
                                  state_count * sizeof(InputIndex),
                                  cudaMemcpyHostToDevice));
        GSC_CUDA_CHECK(cudaMemcpy(d_reach_step,
                                  reach_step_seed.data(),
                                  state_count * sizeof(ReachStep),
                                  cudaMemcpyHostToDevice));
        
        auto init_memcpy_stop = std::chrono::steady_clock::now();
        report.kernel_timings.abstraction_memcpy_h2d_ms = 
            std::chrono::duration<double, std::milli>(init_memcpy_stop - init_memcpy_start).count();

        // 创建CUDA Events用于kernel级别计时
        cudaEvent_t event_start, event_stop;
        GSC_CUDA_CHECK(cudaEventCreate(&event_start));
        GSC_CUDA_CHECK(cudaEventCreate(&event_stop));
        
        auto abstraction_start = std::chrono::steady_clock::now();
        const auto pair_blocks = ceil_div_to_u32(pair_count, threads);
        
        // 构建 GPU 约束参数
        GpuConstraintParams constraint_params;
        constraint_params.constraint_type = static_cast<int>(cfg.constraint_type);
        constraint_params.hyperbolic_a = cfg.hyperbolic_params.a;
        constraint_params.hyperbolic_b = cfg.hyperbolic_params.b;
        constraint_params.hyperbolic_c = cfg.hyperbolic_params.c;
        constraint_params.elliptic_a = cfg.elliptic_params.a;
        constraint_params.elliptic_b = cfg.elliptic_params.b;
        
         std::cout << "GPU: Starting abstraction phase with " << pair_count << " state-input pairs..." << std::endl;
         
         // 构建约束掩码
         std::cout << "GPU: Building constraint mask..." << std::endl;
         const auto state_blocks = ceil_div_to_u32(state_count, threads);
         GSC_CUDA_CHECK(cudaEventRecord(event_start));
         build_constraint_mask_kernel<<<state_blocks, threads>>>(prepared.state_grid,
                                                                  constraint_params,
                                                                  d_constraint_mask);
         GSC_CUDA_CHECK(cudaGetLastError());
         GSC_CUDA_CHECK(cudaEventRecord(event_stop));
         GSC_CUDA_CHECK(cudaEventSynchronize(event_stop));
         float constraint_mask_time = 0.0f;
         GSC_CUDA_CHECK(cudaEventElapsedTime(&constraint_mask_time, event_start, event_stop));
         report.kernel_timings.scatter_mask_ms += static_cast<double>(constraint_mask_time);
         
         // 构建约束前缀和
         std::cout << "GPU: Building constraint prefix sum..." << std::endl;
         build_prefix_on_device(prepared.state_grid, prefix_layout, d_constraint_mask, d_prefix_constraint, report.kernel_timings);
         GSC_CUDA_CHECK(cudaDeviceSynchronize());
         
         // 构建有效掩码前缀和
         std::cout << "GPU: Building valid mask prefix sum..." << std::endl;
         build_prefix_on_device(prepared.state_grid, prefix_layout, d_valid_mask, d_prefix_valid, report.kernel_timings);
         GSC_CUDA_CHECK(cudaDeviceSynchronize());
         
         // 测量抽象kernel耗时
         std::cout << "GPU: Running abstraction kernel..." << std::endl;
         GSC_CUDA_CHECK(cudaEventRecord(event_start));
         abstraction_kernel<<<pair_blocks, threads>>>(prepared.state_grid,
                                                      prepared.input_grid,
                                                      prepared.model,
                                                      constraint_params,
                                                      prefix_layout,
                                                      d_prefix_constraint,
                                                      d_prefix_valid,
                                                      d_min_flat,
                                                      d_max_flat,
                                                      d_pair_valid,
                                                      pair_count);
         GSC_CUDA_CHECK(cudaGetLastError());
         GSC_CUDA_CHECK(cudaEventRecord(event_stop));
         GSC_CUDA_CHECK(cudaEventSynchronize(event_stop));
         float abstraction_kernel_time = 0.0f;
         GSC_CUDA_CHECK(cudaEventElapsedTime(&abstraction_kernel_time, event_start, event_stop));
         report.kernel_timings.abstraction_kernel_ms = static_cast<double>(abstraction_kernel_time);
        
         auto abstraction_stop = std::chrono::steady_clock::now();
         report.abstraction_ms = std::chrono::duration<double, std::milli>(abstraction_stop - abstraction_start).count();
         
         std::cout << "GPU: Abstraction phase completed in " << report.abstraction_ms << " ms" << std::endl;
         std::cout << "  - Kernel execution: " << report.kernel_timings.abstraction_kernel_ms << " ms" << std::endl;
         std::cout << "  - H2D memcpy: " << report.kernel_timings.abstraction_memcpy_h2d_ms << " ms" << std::endl;

         build_prefix_on_device(prepared.state_grid, prefix_layout, d_valid_mask, d_prefix_valid, report.kernel_timings);
         GSC_CUDA_CHECK(cudaDeviceSynchronize());

        auto solve_start = std::chrono::steady_clock::now();
        std::uint64_t certified_candidates = 0;
        std::uint64_t total_reachable = candidate_count;  // 初始可达状态数等于候选数
        
        for (std::uint32_t iter = 1; iter <= cfg.max_iterations; ++iter) {
            auto iter_start = std::chrono::steady_clock::now();
            IterationStats iter_stats;
            iter_stats.iteration = iter;
            
             // 测量前缀和构建耗时
             auto prefix_start = std::chrono::steady_clock::now();
             build_prefix_on_device(prepared.state_grid, prefix_layout, d_reachable_prev, d_prefix_reachable, report.kernel_timings);
             GSC_CUDA_CHECK(cudaDeviceSynchronize());
             auto prefix_stop = std::chrono::steady_clock::now();
             iter_stats.prefix_build_ms = std::chrono::duration<double, std::milli>(prefix_stop - prefix_start).count();

            // 测量满足性检查耗时
            auto satisfaction_start = std::chrono::steady_clock::now();
            pair_satisfaction_kernel<<<pair_blocks, threads>>>(prepared.state_grid,
                                                                prefix_layout,
                                                                d_prefix_reachable,
                                                                d_prefix_valid,
                                                                d_min_flat,
                                                                d_max_flat,
                                                                d_pair_valid,
                                                                d_pair_satisfied,
                                                                pair_count);
            GSC_CUDA_CHECK(cudaGetLastError());
            GSC_CUDA_CHECK(cudaDeviceSynchronize());
            auto satisfaction_stop = std::chrono::steady_clock::now();
            iter_stats.satisfaction_check_ms = std::chrono::duration<double, std::milli>(satisfaction_stop - satisfaction_start).count();

             // 测量输入归约耗时
             auto reduction_start = std::chrono::steady_clock::now();
             GSC_CUDA_CHECK(cudaMemset(d_new_reachable, 0, sizeof(unsigned long long)));
             GSC_CUDA_CHECK(cudaMemset(d_new_candidate_certified, 0, sizeof(unsigned long long)));
             const auto state_blocks = ceil_div_to_u32(state_count, threads);
             reduce_inputs_kernel<<<state_blocks, threads>>>(d_valid_mask,
                                                              d_candidate_mask,
                                                              d_reachable_prev,
                                                              d_reachable_next,
                                                              d_controller,
                                                              d_reach_step,
                                                              d_pair_satisfied,
                                                              state_count,
                                                              static_cast<std::uint32_t>(prepared.input_grid.total_size),
                                                              iter,
                                                              d_new_reachable,
                                                              d_new_candidate_certified);
             GSC_CUDA_CHECK(cudaGetLastError());
             GSC_CUDA_CHECK(cudaDeviceSynchronize());
             auto reduction_kernel_stop = std::chrono::steady_clock::now();
             double reduction_kernel_ms = std::chrono::duration<double, std::milli>(reduction_kernel_stop - reduction_start).count();

             // 测量结果memcpy耗时
             auto memcpy_start = std::chrono::steady_clock::now();
             unsigned long long h_new_reachable = 0;
             unsigned long long h_new_candidate_certified = 0;
             GSC_CUDA_CHECK(cudaMemcpy(&h_new_reachable,
                                        d_new_reachable,
                                        sizeof(unsigned long long),
                                        cudaMemcpyDeviceToHost));
             GSC_CUDA_CHECK(cudaMemcpy(&h_new_candidate_certified,
                                        d_new_candidate_certified,
                                        sizeof(unsigned long long),
                                        cudaMemcpyDeviceToHost));
             auto memcpy_stop = std::chrono::steady_clock::now();
             double iteration_memcpy_ms = std::chrono::duration<double, std::milli>(memcpy_stop - memcpy_start).count();
             
             auto reduction_stop = std::chrono::steady_clock::now();
             iter_stats.reduction_ms = std::chrono::duration<double, std::milli>(reduction_stop - reduction_start).count();

             // 更新统计信息
             certified_candidates += h_new_candidate_certified;
             total_reachable += h_new_reachable;
             iter_stats.newly_reachable = h_new_reachable;
             iter_stats.newly_certified = h_new_candidate_certified;
             iter_stats.total_reachable = total_reachable;
             iter_stats.total_certified = certified_candidates;
             
             // 累加kernel计时
             report.kernel_timings.pair_satisfaction_ms += iter_stats.satisfaction_check_ms;
             report.kernel_timings.reduce_inputs_ms += reduction_kernel_ms;
             report.kernel_timings.prefix_build_total_ms += iter_stats.prefix_build_ms;
             report.kernel_timings.iteration_memcpy_ms += iteration_memcpy_ms;
             
             auto iter_stop = std::chrono::steady_clock::now();
             iter_stats.iteration_ms = std::chrono::duration<double, std::milli>(iter_stop - iter_start).count();
             
             // 保存迭代统计
             report.iteration_stats.push_back(iter_stats);
             
             std::swap(d_reachable_prev, d_reachable_next);
             report.result.iterations = iter;
             
             // Print progress every 10 iterations or if it's the first/last few iterations
             if (iter <= 5 || iter % 10 == 0 || iter == cfg.max_iterations) {
                 std::cout << "GPU: Iteration " << iter << "/" << cfg.max_iterations 
                           << ", newly reachable: " << h_new_reachable 
                           << ", newly certified: " << h_new_candidate_certified
                           << ", total certified: " << certified_candidates << "/" << candidate_count
                           << " (" << iter_stats.iteration_ms << " ms)"
                           << " [prefix: " << iter_stats.prefix_build_ms << " ms"
                           << ", satisfaction: " << iter_stats.satisfaction_check_ms << " ms"
                           << ", reduction: " << reduction_kernel_ms << " ms]"
                           << std::endl;
             }

            if (certified_candidates == candidate_count) {
                report.result.converged = true;
                report.result.message = "all candidate states obtained a controller";
                break;
            }
            if (h_new_reachable == 0 && h_new_candidate_certified == 0) {
                report.result.message = "iteration stalled before candidate closure";
                break;
            }
        }
        auto solve_stop = std::chrono::steady_clock::now();
        report.solve_ms = std::chrono::duration<double, std::milli>(solve_stop - solve_start).count();
        
         std::cout << "GPU: Solve phase completed in " << report.solve_ms << " ms" << std::endl;
         std::cout << "GPU: Final status - converged: " << (report.result.converged ? "true" : "false") 
                   << ", iterations: " << report.result.iterations
                   << ", message: " << report.result.message << std::endl;

         // 测量结果复制耗时
         auto result_copy_start = std::chrono::steady_clock::now();
         
         // 复制结果数据到主机
         report.result.valid_mask.resize(state_count);
         report.result.candidate_mask.resize(state_count);
         report.result.reachable_mask.resize(state_count);
         report.result.controller.resize(state_count);
         report.result.reach_step.resize(state_count);
         GSC_CUDA_CHECK(cudaMemcpy(report.result.valid_mask.data(),
                                   d_valid_mask,
                                   state_count * sizeof(std::uint8_t),
                                   cudaMemcpyDeviceToHost));
         GSC_CUDA_CHECK(cudaMemcpy(report.result.candidate_mask.data(),
                                   d_candidate_mask,
                                   state_count * sizeof(std::uint8_t),
                                   cudaMemcpyDeviceToHost));
         GSC_CUDA_CHECK(cudaMemcpy(report.result.reachable_mask.data(),
                                   d_reachable_prev,
                                   state_count * sizeof(std::uint8_t),
                                   cudaMemcpyDeviceToHost));
         GSC_CUDA_CHECK(cudaMemcpy(report.result.controller.data(),
                                   d_controller,
                                   state_count * sizeof(InputIndex),
                                   cudaMemcpyDeviceToHost));
         GSC_CUDA_CHECK(cudaMemcpy(report.result.reach_step.data(),
                                   d_reach_step,
                                   state_count * sizeof(ReachStep),
                                   cudaMemcpyDeviceToHost));

         // 复制抽象数据到主机（用于分析和可视化）
         std::cout << "GPU: Copying abstraction data to host (" 
                   << (pair_count * (2 * sizeof(std::uint32_t) + sizeof(std::uint8_t)) / (1024.0 * 1024.0))
                   << " MB)..." << std::endl;
         report.abstraction_min_flat.resize(pair_count);
         report.abstraction_max_flat.resize(pair_count);
         report.abstraction_valid.resize(pair_count);
         GSC_CUDA_CHECK(cudaMemcpy(report.abstraction_min_flat.data(),
                                   d_min_flat,
                                   pair_count * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost));
         GSC_CUDA_CHECK(cudaMemcpy(report.abstraction_max_flat.data(),
                                   d_max_flat,
                                   pair_count * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost));
         GSC_CUDA_CHECK(cudaMemcpy(report.abstraction_valid.data(),
                                   d_pair_valid,
                                   pair_count * sizeof(std::uint8_t),
                                   cudaMemcpyDeviceToHost));
         
         auto result_copy_stop = std::chrono::steady_clock::now();
         report.kernel_timings.result_memcpy_d2h_ms = 
             std::chrono::duration<double, std::milli>(result_copy_stop - result_copy_start).count();

         report.result.reachable_states =
             std::count(report.result.reachable_mask.begin(), report.result.reachable_mask.end(), std::uint8_t{1});
         report.result.certified_candidate_states = certified_candidates;
         report.result.solve_ms = report.solve_ms;
         report.executed = true;
         if (!report.result.converged && report.result.message.empty()) {
             report.result.message = "maximum iterations reached";
         }
         
         // 计算总计时统计
         report.total_ms = report.abstraction_ms + report.solve_ms;
         report.kernel_timings.total_kernel_ms = 
             report.kernel_timings.abstraction_kernel_ms +
             report.kernel_timings.pair_satisfaction_ms +
             report.kernel_timings.reduce_inputs_ms +
             report.kernel_timings.prefix_build_total_ms;
         report.kernel_timings.total_memcpy_ms = 
             report.kernel_timings.abstraction_memcpy_h2d_ms +
             report.kernel_timings.iteration_memcpy_ms +
             report.kernel_timings.result_memcpy_d2h_ms;
         report.kernel_timings.total_compute_ms = 
             report.kernel_timings.total_kernel_ms + report.kernel_timings.total_memcpy_ms;
         
         // 清理CUDA Events
         cudaEventDestroy(event_start);
         cudaEventDestroy(event_stop);
    } catch (const std::exception& ex) {
        report.message = ex.what();
    }

    cudaFree(d_min_flat);
    cudaFree(d_max_flat);
    cudaFree(d_pair_valid);
    cudaFree(d_pair_satisfied);
    cudaFree(d_valid_mask);
    cudaFree(d_candidate_mask);
    cudaFree(d_constraint_mask);
    cudaFree(d_reachable_prev);
    cudaFree(d_reachable_next);
    cudaFree(d_controller);
    cudaFree(d_reach_step);
    cudaFree(d_prefix_valid);
    cudaFree(d_prefix_constraint);
    cudaFree(d_prefix_reachable);
    cudaFree(d_new_reachable);
    cudaFree(d_new_candidate_certified);
    return report;
}

}  // namespace gsc

#endif
