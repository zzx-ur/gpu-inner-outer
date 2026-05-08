#include "gsc/result_io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "gsc/types.hpp"
#include "gsc/cuda_api.hpp"

#if GSC_HAS_HDF5
#include <hdf5.h>
#endif

namespace gsc {

namespace {

std::string sanitize_token(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char ch : input) {
        if (std::isalnum(ch) || ch == '_' || ch == '-') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) {
        return "unnamed";
    }
    return out;
}

#if GSC_HAS_HDF5

template <typename CloseFn>
class H5Handle {
public:
    H5Handle() = default;

    H5Handle(hid_t id, CloseFn closer)
        : id_(id), closer_(closer) {}

    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;

    H5Handle(H5Handle&& other) noexcept
        : id_(other.id_), closer_(other.closer_) {
        other.id_ = -1;
    }

    H5Handle& operator=(H5Handle&& other) noexcept {
        if (this != &other) {
            reset();
            id_ = other.id_;
            closer_ = other.closer_;
            other.id_ = -1;
        }
        return *this;
    }

    ~H5Handle() {
        reset();
    }

    hid_t get() const {
        return id_;
    }

    explicit operator bool() const {
        return id_ >= 0;
    }

private:
    void reset() {
        if (id_ >= 0) {
            closer_(id_);
            id_ = -1;
        }
    }

    hid_t id_ = -1;
    CloseFn closer_ = nullptr;
};

using FileHandle = H5Handle<herr_t (*)(hid_t)>;
using GroupHandle = H5Handle<herr_t (*)(hid_t)>;
using SpaceHandle = H5Handle<herr_t (*)(hid_t)>;
using DatasetHandle = H5Handle<herr_t (*)(hid_t)>;
using TypeHandle = H5Handle<herr_t (*)(hid_t)>;

void expect_h5(herr_t status, const std::string& message) {
    if (status < 0) {
        throw std::runtime_error(message);
    }
}

template <typename HandleT>
hid_t require_id(const HandleT& handle, const std::string& message) {
    if (!handle) {
        throw std::runtime_error(message);
    }
    return handle.get();
}

GroupHandle create_group(hid_t parent, const std::string& name) {
    auto group = GroupHandle(H5Gcreate2(parent, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), &H5Gclose);
    require_id(group, "failed to create HDF5 group: " + name);
    return group;
}

template <typename T>
hid_t hdf5_native_type();

template <>
hid_t hdf5_native_type<double>() {
    return H5T_NATIVE_DOUBLE;
}

template <>
hid_t hdf5_native_type<std::uint8_t>() {
    return H5T_NATIVE_UINT8;
}

template <>
hid_t hdf5_native_type<std::uint32_t>() {
    return H5T_NATIVE_UINT32;
}

template <>
hid_t hdf5_native_type<std::uint64_t>() {
    return H5T_NATIVE_UINT64;
}

template <>
hid_t hdf5_native_type<int>() {
    return H5T_NATIVE_INT;
}

template <typename T>
void write_scalar(hid_t parent, const std::string& name, const T& value) {
    auto space = SpaceHandle(H5Screate(H5S_SCALAR), &H5Sclose);
    require_id(space, "failed to create scalar dataspace for " + name);

    auto dataset = DatasetHandle(
        H5Dcreate2(parent, name.c_str(), hdf5_native_type<T>(), space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        &H5Dclose);
    require_id(dataset, "failed to create dataset " + name);
    expect_h5(H5Dwrite(dataset.get(), hdf5_native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &value),
              "failed to write dataset " + name);
}

template <typename T>
void write_vector(hid_t parent, const std::string& name, const T* data, std::size_t size) {
    const hsize_t dims[1] = {static_cast<hsize_t>(size)};
    auto space = SpaceHandle(H5Screate_simple(1, dims, nullptr), &H5Sclose);
    require_id(space, "failed to create dataspace for " + name);

    auto dataset = DatasetHandle(
        H5Dcreate2(parent, name.c_str(), hdf5_native_type<T>(), space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        &H5Dclose);
    require_id(dataset, "failed to create dataset " + name);
    expect_h5(H5Dwrite(dataset.get(), hdf5_native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
              "failed to write dataset " + name);
}

template <typename T, std::size_t N>
void write_array(hid_t parent, const std::string& name, const T (&data)[N]) {
    write_vector(parent, name, data, N);
}

template <typename T>
void write_vector(hid_t parent, const std::string& name, const std::vector<T>& data) {
    write_vector(parent, name, data.data(), data.size());
}

template <typename T>
void write_matrix(hid_t parent, const std::string& name, const T* data, hsize_t rows, hsize_t cols) {
    const hsize_t dims[2] = {rows, cols};
    auto space = SpaceHandle(H5Screate_simple(2, dims, nullptr), &H5Sclose);
    require_id(space, "failed to create matrix dataspace for " + name);

    auto dataset = DatasetHandle(
        H5Dcreate2(parent, name.c_str(), hdf5_native_type<T>(), space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        &H5Dclose);
    require_id(dataset, "failed to create matrix dataset " + name);
    expect_h5(H5Dwrite(dataset.get(), hdf5_native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
              "failed to write matrix dataset " + name);
}

void write_string(hid_t parent, const std::string& name, const std::string& value) {
    auto type = TypeHandle(H5Tcopy(H5T_C_S1), &H5Tclose);
    require_id(type, "failed to create string type for " + name);
    const std::size_t length = std::max<std::size_t>(1, value.size() + 1);
    expect_h5(H5Tset_size(type.get(), length), "failed to size string type for " + name);
    expect_h5(H5Tset_strpad(type.get(), H5T_STR_NULLTERM), "failed to configure string type for " + name);

    auto space = SpaceHandle(H5Screate(H5S_SCALAR), &H5Sclose);
    require_id(space, "failed to create string dataspace for " + name);

    auto dataset =
        DatasetHandle(H5Dcreate2(parent, name.c_str(), type.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                      &H5Dclose);
    require_id(dataset, "failed to create string dataset " + name);

    const char* ptr = value.empty() ? "" : value.c_str();
    expect_h5(H5Dwrite(dataset.get(), type.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr),
              "failed to write string dataset " + name);
}

std::vector<double> build_input_centers(const InputGrid2D& input_grid) {
    std::vector<double> centers(input_grid.total_size * kInputDim, 0.0);
    for (std::uint64_t flat = 0; flat < input_grid.total_size; ++flat) {
        double center[kInputDim];
        input_grid.center(static_cast<InputIndex>(flat), center);
        for (int dim = 0; dim < kInputDim; ++dim) {
            centers[flat * kInputDim + dim] = center[dim];
        }
    }
    return centers;
}

std::vector<double> build_obstacle_matrix(const std::vector<HyperRect4D>& obstacles) {
    std::vector<double> data;
    data.reserve(obstacles.size() * 2 * kStateDim);
    for (const auto& obstacle : obstacles) {
        data.insert(data.end(), obstacle.lb, obstacle.lb + kStateDim);
        data.insert(data.end(), obstacle.ub, obstacle.ub + kStateDim);
    }
    return data;
}

void write_common_metadata(hid_t file_id,
                           const CaseConfig& cfg,
                           const PreparedCase<std::uint32_t>& prepared,
                           const std::vector<ResultExportView>& runs) {
    auto meta = create_group(file_id, "meta");
    auto state = create_group(file_id, "state");
    auto input = create_group(file_id, "input");
    auto config = create_group(file_id, "config");
    auto summary = create_group(file_id, "summary");
    auto result = create_group(file_id, "result");

    write_string(meta.get(), "case_name", cfg.name);
    write_scalar(meta.get(), "invalid_input", kInvalidInput);
    write_scalar(meta.get(), "unreachable_step", kUnreachableStep);
    write_scalar(meta.get(), "run_count", static_cast<std::uint32_t>(runs.size()));

    write_array(state.get(), "lb", prepared.state_grid.lb);
    write_array(state.get(), "ub", prepared.state_grid.ub);
    write_array(state.get(), "eta", prepared.state_grid.eta);
    write_array(state.get(), "shape", prepared.state_grid.shape);
    write_array(state.get(), "strides", prepared.state_grid.strides);
    write_scalar(state.get(), "total_size", prepared.state_grid.total_size);
    write_scalar(state.get(), "wrap_dim", prepared.state_grid.wrap_dim);

    write_array(input.get(), "lb", prepared.input_grid.lb);
    write_array(input.get(), "ub", prepared.input_grid.ub);
    write_array(input.get(), "eta", prepared.input_grid.eta);
    write_array(input.get(), "shape", prepared.input_grid.shape);
    write_array(input.get(), "strides", prepared.input_grid.strides);
    write_scalar(input.get(), "total_size", prepared.input_grid.total_size);
    const auto input_centers = build_input_centers(prepared.input_grid);
    write_matrix(input.get(), "centers", input_centers.data(), prepared.input_grid.total_size, kInputDim);

    write_array(config.get(), "candidate_lb", cfg.candidate.lb);
    write_array(config.get(), "candidate_ub", cfg.candidate.ub);
    write_array(config.get(), "map_lb", cfg.map.lb);
    write_array(config.get(), "map_ub", cfg.map.ub);
    write_array(config.get(), "disturbance_half_width", cfg.disturbance_half_width);
    write_scalar(config.get(), "sample_time", cfg.sample_time);
    write_scalar(config.get(), "integration_substeps", cfg.integration_substeps);
    write_scalar(config.get(), "max_iterations", cfg.max_iterations);
    write_scalar(config.get(), "verbose", static_cast<std::uint8_t>(cfg.verbose ? 1 : 0));

    const auto obstacle_data = build_obstacle_matrix(cfg.obstacles);
    if (obstacle_data.empty()) {
        const hsize_t dims[2] = {0, static_cast<hsize_t>(2 * kStateDim)};
        auto space = SpaceHandle(H5Screate_simple(2, dims, nullptr), &H5Sclose);
        require_id(space, "failed to create obstacle dataspace");
        auto dataset = DatasetHandle(
            H5Dcreate2(config.get(), "obstacles", H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            &H5Dclose);
        require_id(dataset, "failed to create obstacle dataset");
    } else {
        write_matrix(config.get(), "obstacles", obstacle_data.data(), cfg.obstacles.size(), 2 * kStateDim);
    }

    for (const auto& run : runs) {
        if (!run.result) {
            throw std::runtime_error("result export view has null result pointer");
        }

        auto summary_group = create_group(summary.get(), sanitize_token(run.label));
        auto result_group = create_group(result.get(), sanitize_token(run.label));

        write_scalar(summary_group.get(), "iterations", run.result->iterations);
        write_scalar(summary_group.get(), "converged", static_cast<std::uint8_t>(run.result->converged ? 1 : 0));
        write_string(summary_group.get(), "message", run.result->message);
        write_scalar(summary_group.get(), "reachable_states", run.result->reachable_states);
        write_scalar(summary_group.get(), "certified_candidate_states", run.result->certified_candidate_states);
        write_scalar(summary_group.get(), "abstraction_ms", prepared.abstraction_ms);
        write_scalar(summary_group.get(), "solve_ms", run.result->solve_ms);

        write_vector(result_group.get(), "valid_mask", run.result->valid_mask);
        write_vector(result_group.get(), "candidate_mask", run.result->candidate_mask);
        write_vector(result_group.get(), "reachable_mask", run.result->reachable_mask);
        write_vector(result_group.get(), "controller", run.result->controller);
        write_vector(result_group.get(), "reach_step", run.result->reach_step);
    }
}

#endif

}  // namespace

bool hdf5_export_compiled() {
#if GSC_HAS_HDF5
    return true;
#else
    return false;
#endif
}

std::string default_result_path(const CaseConfig& cfg, const std::string& mode) {
    const std::filesystem::path dir("results");
    const std::string safe_name = sanitize_token(cfg.name);
    const std::string safe_mode = sanitize_token(mode);
    return (dir / (safe_name + "_" + safe_mode + ".h5")).string();
}

void write_results_hdf5(const std::string& output_path,
                        const CaseConfig& cfg,
                        const PreparedCase<std::uint32_t>& prepared,
                        const std::vector<ResultExportView>& runs) {
#if GSC_HAS_HDF5
    if (runs.empty()) {
        throw std::runtime_error("no runs were provided for HDF5 export");
    }

    const auto output = std::filesystem::path(output_path);
    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }

    auto file = FileHandle(H5Fcreate(output.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), &H5Fclose);
    require_id(file, "failed to create HDF5 file: " + output.string());

    write_common_metadata(file.get(), cfg, prepared, runs);
#else
    (void)output_path;
    (void)cfg;
    (void)prepared;
    (void)runs;
    throw std::runtime_error("HDF5 export support was not compiled");
#endif
}

void write_gpu_results_hdf5(const std::string& output_path,
                            const CaseConfig& cfg,
                            const GpuRunReport& gpu_report) {
#if GSC_HAS_HDF5
    if (!gpu_report.executed) {
        throw std::runtime_error("GPU report has not been executed");
    }

    const auto output = std::filesystem::path(output_path);
    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }

    auto file = FileHandle(H5Fcreate(output.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), &H5Fclose);
    require_id(file, "failed to create HDF5 file: " + output.string());

    // 创建基本组
    auto meta = create_group(file.get(), "meta");
    auto state = create_group(file.get(), "state");
    auto input = create_group(file.get(), "input");
    auto config = create_group(file.get(), "config");
    auto summary = create_group(file.get(), "summary");
    auto result = create_group(file.get(), "result");
    auto abstraction = create_group(file.get(), "abstraction");
    auto iterations = create_group(file.get(), "iterations");
    auto timings = create_group(file.get(), "timings");

    // 写入元数据
    write_string(meta.get(), "case_name", cfg.name);
    write_scalar(meta.get(), "invalid_input", kInvalidInput);
    write_scalar(meta.get(), "unreachable_step", kUnreachableStep);
    write_scalar(meta.get(), "run_count", static_cast<std::uint32_t>(1));

    // 写入状态网格信息
    write_array(state.get(), "lb", gpu_report.state_grid.lb);
    write_array(state.get(), "ub", gpu_report.state_grid.ub);
    write_array(state.get(), "eta", gpu_report.state_grid.eta);
    write_array(state.get(), "shape", gpu_report.state_grid.shape);
    write_array(state.get(), "strides", gpu_report.state_grid.strides);
    write_scalar(state.get(), "total_size", gpu_report.state_grid.total_size);
    write_scalar(state.get(), "wrap_dim", gpu_report.state_grid.wrap_dim);

    // 写入输入网格信息
    write_array(input.get(), "lb", gpu_report.input_grid.lb);
    write_array(input.get(), "ub", gpu_report.input_grid.ub);
    write_array(input.get(), "eta", gpu_report.input_grid.eta);
    write_array(input.get(), "shape", gpu_report.input_grid.shape);
    write_array(input.get(), "strides", gpu_report.input_grid.strides);
    write_scalar(input.get(), "total_size", gpu_report.input_grid.total_size);
    const auto input_centers = build_input_centers(gpu_report.input_grid);
    write_matrix(input.get(), "centers", input_centers.data(), gpu_report.input_grid.total_size, kInputDim);

    // 写入配置信息
    write_array(config.get(), "candidate_lb", cfg.candidate.lb);
    write_array(config.get(), "candidate_ub", cfg.candidate.ub);
    write_array(config.get(), "map_lb", cfg.map.lb);
    write_array(config.get(), "map_ub", cfg.map.ub);
    write_array(config.get(), "disturbance_half_width", cfg.disturbance_half_width);
    write_scalar(config.get(), "sample_time", cfg.sample_time);
    write_scalar(config.get(), "integration_substeps", cfg.integration_substeps);
    write_scalar(config.get(), "max_iterations", cfg.max_iterations);
    write_scalar(config.get(), "verbose", static_cast<std::uint8_t>(cfg.verbose ? 1 : 0));

    const auto obstacle_data = build_obstacle_matrix(cfg.obstacles);
    if (obstacle_data.empty()) {
        const hsize_t dims[2] = {0, static_cast<hsize_t>(2 * kStateDim)};
        auto space = SpaceHandle(H5Screate_simple(2, dims, nullptr), &H5Sclose);
        require_id(space, "failed to create obstacle dataspace");
        auto dataset = DatasetHandle(
            H5Dcreate2(config.get(), "obstacles", H5T_NATIVE_DOUBLE, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
            &H5Dclose);
        require_id(dataset, "failed to create obstacle dataset");
    } else {
        write_matrix(config.get(), "obstacles", obstacle_data.data(), cfg.obstacles.size(), 2 * kStateDim);
    }

    // 写入汇总信息
    auto summary_gpu = create_group(summary.get(), "gpu");
    write_scalar(summary_gpu.get(), "iterations", gpu_report.result.iterations);
    write_scalar(summary_gpu.get(), "converged", static_cast<std::uint8_t>(gpu_report.result.converged ? 1 : 0));
    write_string(summary_gpu.get(), "message", gpu_report.result.message);
    write_scalar(summary_gpu.get(), "reachable_states", gpu_report.result.reachable_states);
    write_scalar(summary_gpu.get(), "certified_candidate_states", gpu_report.result.certified_candidate_states);
    write_scalar(summary_gpu.get(), "abstraction_ms", gpu_report.abstraction_ms);
    write_scalar(summary_gpu.get(), "solve_ms", gpu_report.solve_ms);
    write_scalar(summary_gpu.get(), "pair_count", gpu_report.pair_count);

    // 写入结果数据
    auto result_gpu = create_group(result.get(), "gpu");
    write_vector(result_gpu.get(), "valid_mask", gpu_report.result.valid_mask);
    write_vector(result_gpu.get(), "candidate_mask", gpu_report.result.candidate_mask);
    write_vector(result_gpu.get(), "reachable_mask", gpu_report.result.reachable_mask);
    write_vector(result_gpu.get(), "controller", gpu_report.result.controller);
    write_vector(result_gpu.get(), "reach_step", gpu_report.result.reach_step);

    // 写入抽象数据（后继盒信息）
    write_scalar(abstraction.get(), "pair_count", gpu_report.pair_count);
    write_vector(abstraction.get(), "min_flat", gpu_report.abstraction_min_flat);
    write_vector(abstraction.get(), "max_flat", gpu_report.abstraction_max_flat);
    write_vector(abstraction.get(), "valid", gpu_report.abstraction_valid);

    // 写入逐迭代统计
    if (!gpu_report.iteration_stats.empty()) {
        const std::size_t n_iters = gpu_report.iteration_stats.size();
        std::vector<std::uint32_t> iter_nums(n_iters);
        std::vector<std::uint64_t> newly_reachable(n_iters);
        std::vector<std::uint64_t> newly_certified(n_iters);
        std::vector<std::uint64_t> total_reachable(n_iters);
        std::vector<std::uint64_t> total_certified(n_iters);
        std::vector<double> iteration_ms(n_iters);
        std::vector<double> prefix_build_ms(n_iters);
        std::vector<double> satisfaction_check_ms(n_iters);
        std::vector<double> reduction_ms(n_iters);

        for (std::size_t i = 0; i < n_iters; ++i) {
            const auto& stats = gpu_report.iteration_stats[i];
            iter_nums[i] = stats.iteration;
            newly_reachable[i] = stats.newly_reachable;
            newly_certified[i] = stats.newly_certified;
            total_reachable[i] = stats.total_reachable;
            total_certified[i] = stats.total_certified;
            iteration_ms[i] = stats.iteration_ms;
            prefix_build_ms[i] = stats.prefix_build_ms;
            satisfaction_check_ms[i] = stats.satisfaction_check_ms;
            reduction_ms[i] = stats.reduction_ms;
        }

        write_vector(iterations.get(), "iteration", iter_nums);
        write_vector(iterations.get(), "newly_reachable", newly_reachable);
        write_vector(iterations.get(), "newly_certified", newly_certified);
        write_vector(iterations.get(), "total_reachable", total_reachable);
        write_vector(iterations.get(), "total_certified", total_certified);
        write_vector(iterations.get(), "iteration_ms", iteration_ms);
        write_vector(iterations.get(), "prefix_build_ms", prefix_build_ms);
        write_vector(iterations.get(), "satisfaction_check_ms", satisfaction_check_ms);
        write_vector(iterations.get(), "reduction_ms", reduction_ms);
    }

    // 写入kernel级别计时
    write_scalar(timings.get(), "abstraction_kernel_ms", gpu_report.kernel_timings.abstraction_kernel_ms);
    write_scalar(timings.get(), "scatter_mask_ms", gpu_report.kernel_timings.scatter_mask_ms);
    write_scalar(timings.get(), "scan_axis0_ms", gpu_report.kernel_timings.scan_axis0_ms);
    write_scalar(timings.get(), "scan_axis1_ms", gpu_report.kernel_timings.scan_axis1_ms);
    write_scalar(timings.get(), "scan_axis2_ms", gpu_report.kernel_timings.scan_axis2_ms);
    write_scalar(timings.get(), "scan_axis3_ms", gpu_report.kernel_timings.scan_axis3_ms);
    write_scalar(timings.get(), "pair_satisfaction_ms", gpu_report.kernel_timings.pair_satisfaction_ms);
    write_scalar(timings.get(), "reduce_inputs_ms", gpu_report.kernel_timings.reduce_inputs_ms);
    write_scalar(timings.get(), "memcpy_h2d_ms", gpu_report.kernel_timings.memcpy_h2d_ms);
    write_scalar(timings.get(), "memcpy_d2h_ms", gpu_report.kernel_timings.memcpy_d2h_ms);

#else
    (void)output_path;
    (void)cfg;
    (void)gpu_report;
    throw std::runtime_error("HDF5 export support was not compiled");
#endif
}

}  // namespace gsc
