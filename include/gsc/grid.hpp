#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "gsc/types.hpp"

namespace gsc {

template <typename StateIndex>
struct Grid4D {
    double lb[kStateDim]{};
    double ub[kStateDim]{};
    double eta[kStateDim]{};
    std::uint32_t shape[kStateDim]{};
    std::uint64_t strides[kStateDim]{};
    std::uint64_t total_size = 0;
    int wrap_dim = 2;

    GSC_HD StateIndex flatten(const std::uint32_t idx[kStateDim]) const {
        return static_cast<StateIndex>(
            static_cast<std::uint64_t>(idx[0]) * strides[0] +
            static_cast<std::uint64_t>(idx[1]) * strides[1] +
            static_cast<std::uint64_t>(idx[2]) * strides[2] +
            static_cast<std::uint64_t>(idx[3]) * strides[3]);
    }

    GSC_HD void unflatten(StateIndex flat, std::uint32_t idx[kStateDim]) const {
        std::uint64_t value = static_cast<std::uint64_t>(flat);
        for (int dim = kStateDim - 1; dim >= 0; --dim) {
            idx[dim] = static_cast<std::uint32_t>(value / strides[dim]);
            value %= strides[dim];
        }
    }

    GSC_HD void center(StateIndex flat, double out[kStateDim]) const {
        std::uint32_t idx[kStateDim];
        unflatten(flat, idx);
        for (int dim = 0; dim < kStateDim; ++dim) {
            out[dim] = lb[dim] + (static_cast<double>(idx[dim]) + 0.5) * eta[dim];
        }
    }

    GSC_HD double wrap_value(double x) const {
        const double low = lb[wrap_dim];
        const double high = ub[wrap_dim];
        const double span = high - low;
        while (x < low) {
            x += span;
        }
        while (x >= high) {
            x -= span;
        }
        return x;
    }
};

struct InputGrid2D {
    double lb[kInputDim]{};
    double ub[kInputDim]{};
    double eta[kInputDim]{};
    std::uint32_t shape[kInputDim]{};
    std::uint64_t strides[kInputDim]{};
    std::uint64_t total_size = 0;

    GSC_HD void center(InputIndex flat, double out[kInputDim]) const {
        std::uint64_t value = flat;
        std::uint32_t idx[kInputDim]{};
        for (int dim = kInputDim - 1; dim >= 0; --dim) {
            idx[dim] = static_cast<std::uint32_t>(value / strides[dim]);
            value %= strides[dim];
        }
        for (int dim = 0; dim < kInputDim; ++dim) {
            out[dim] = lb[dim] + (static_cast<double>(idx[dim]) + 0.5) * eta[dim];
        }
    }
};

struct IndexBox4D {
    std::uint32_t min[kStateDim]{};
    std::uint32_t max[kStateDim]{};
    bool valid = false;
};

struct PrefixLayout4D {
    std::uint32_t padded_shape[kStateDim]{};
    std::uint64_t strides[kStateDim]{};
    std::uint64_t total_size = 0;

    GSC_HD std::uint64_t offset(std::uint32_t x0,
                                std::uint32_t x1,
                                std::uint32_t x2,
                                std::uint32_t x3) const {
        return static_cast<std::uint64_t>(x0) * strides[0] +
               static_cast<std::uint64_t>(x1) * strides[1] +
               static_cast<std::uint64_t>(x2) * strides[2] +
               static_cast<std::uint64_t>(x3) * strides[3];
    }
};

template <typename StateIndex>
inline Grid4D<StateIndex> build_state_grid(const double lb[kStateDim],
                                           const double ub[kStateDim],
                                           const double eta[kStateDim],
                                           int wrap_dim) {
    Grid4D<StateIndex> grid;
    grid.wrap_dim = wrap_dim;

    std::uint64_t total = 1;
    for (int dim = 0; dim < kStateDim; ++dim) {
        grid.lb[dim] = lb[dim];
        grid.ub[dim] = ub[dim];
        grid.eta[dim] = eta[dim];
        if (!(eta[dim] > 0.0)) {
            throw std::runtime_error("state eta must be positive");
        }
        const double width = ub[dim] - lb[dim];
        if (!(width > 0.0)) {
            throw std::runtime_error("state bounds must satisfy ub > lb");
        }
        grid.shape[dim] = static_cast<std::uint32_t>(std::ceil(width / eta[dim]));
    }

    grid.strides[0] = 1;
    for (int dim = 1; dim < kStateDim; ++dim) {
        grid.strides[dim] = grid.strides[dim - 1] * grid.shape[dim - 1];
    }

    for (int dim = 0; dim < kStateDim; ++dim) {
        total *= grid.shape[dim];
    }
    grid.total_size = total;
    if (requires_u64_index<StateIndex>(grid.total_size)) {
        throw std::runtime_error("state grid exceeds the selected state index type");
    }
    return grid;
}

inline InputGrid2D build_input_grid(const double lb[kInputDim],
                                    const double ub[kInputDim],
                                    const double eta[kInputDim]) {
    InputGrid2D grid;
    std::uint64_t total = 1;
    for (int dim = 0; dim < kInputDim; ++dim) {
        grid.lb[dim] = lb[dim];
        grid.ub[dim] = ub[dim];
        grid.eta[dim] = eta[dim];
        if (!(eta[dim] > 0.0)) {
            throw std::runtime_error("input eta must be positive");
        }
        const double width = ub[dim] - lb[dim];
        if (!(width > 0.0)) {
            throw std::runtime_error("input bounds must satisfy ub > lb");
        }
        grid.shape[dim] = static_cast<std::uint32_t>(std::ceil(width / eta[dim]));
    }

    grid.strides[0] = 1;
    grid.strides[1] = grid.shape[0];
    total = static_cast<std::uint64_t>(grid.shape[0]) * grid.shape[1];
    grid.total_size = total;
    if (requires_u64_index<InputIndex>(grid.total_size)) {
        throw std::runtime_error("input grid exceeds uint32_t");
    }
    return grid;
}

template <typename StateIndex>
GSC_HD inline bool index_box_from_flats(const Grid4D<StateIndex>& grid,
                                        StateIndex min_flat,
                                        StateIndex max_flat,
                                        IndexBox4D* out) {
    grid.unflatten(min_flat, out->min);
    grid.unflatten(max_flat, out->max);
    out->valid = true;
    return true;
}

template <typename StateIndex>
GSC_HD inline bool continuous_box_to_index_box(const Grid4D<StateIndex>& grid,
                                               const double min_coord[kStateDim],
                                               const double max_coord[kStateDim],
                                               IndexBox4D* out) {
    constexpr double kEps = 1e-9;
    out->valid = false;

    for (int dim = 0; dim < kStateDim; ++dim) {
        if (dim == grid.wrap_dim) {
            continue;
        }
        if (min_coord[dim] < grid.lb[dim] - kEps || max_coord[dim] > grid.ub[dim] + kEps) {
            return false;
        }
    }

    const double wrap_span = grid.ub[grid.wrap_dim] - grid.lb[grid.wrap_dim];
    const double raw_wrap_width = max_coord[grid.wrap_dim] - min_coord[grid.wrap_dim];

    for (int dim = 0; dim < kStateDim; ++dim) {
        if (dim == grid.wrap_dim) {
            if (raw_wrap_width >= wrap_span - kEps) {
                out->min[dim] = 0;
                out->max[dim] = grid.shape[dim] - 1;
                continue;
            }

            const double min_wrapped = grid.wrap_value(min_coord[dim]);
            const double max_wrapped = grid.wrap_value(max_coord[dim]);

            auto lower = static_cast<std::int64_t>(std::floor((min_wrapped - grid.lb[dim]) / grid.eta[dim]));
            auto upper = static_cast<std::int64_t>(std::ceil((max_wrapped - grid.lb[dim]) / grid.eta[dim])) - 1;
            if (lower < 0) {
                lower = 0;
            }
            if (upper < 0) {
                upper = 0;
            }
            if (lower >= static_cast<std::int64_t>(grid.shape[dim])) {
                lower = static_cast<std::int64_t>(grid.shape[dim]) - 1;
            }
            if (upper >= static_cast<std::int64_t>(grid.shape[dim])) {
                upper = static_cast<std::int64_t>(grid.shape[dim]) - 1;
            }
            out->min[dim] = static_cast<std::uint32_t>(lower);
            out->max[dim] = static_cast<std::uint32_t>(upper);
            continue;
        }

        auto lower = static_cast<std::int64_t>(std::floor((min_coord[dim] - grid.lb[dim]) / grid.eta[dim]));
        auto upper = static_cast<std::int64_t>(std::ceil((max_coord[dim] - grid.lb[dim]) / grid.eta[dim])) - 1;
        if (lower < 0) {
            lower = 0;
        }
        if (upper < 0) {
            upper = 0;
        }
        if (lower >= static_cast<std::int64_t>(grid.shape[dim])) {
            lower = static_cast<std::int64_t>(grid.shape[dim]) - 1;
        }
        if (upper >= static_cast<std::int64_t>(grid.shape[dim])) {
            upper = static_cast<std::int64_t>(grid.shape[dim]) - 1;
        }
        out->min[dim] = static_cast<std::uint32_t>(lower);
        out->max[dim] = static_cast<std::uint32_t>(upper);
    }

    out->valid = true;
    return true;
}

template <typename StateIndex>
inline PrefixLayout4D build_prefix_layout(const Grid4D<StateIndex>& grid) {
    PrefixLayout4D layout;
    layout.padded_shape[0] = grid.shape[0] + 1;
    layout.padded_shape[1] = grid.shape[1] + 1;
    layout.padded_shape[2] = grid.shape[2] + 1;
    layout.padded_shape[3] = grid.shape[3] + 1;
    layout.strides[0] = 1;
    for (int dim = 1; dim < kStateDim; ++dim) {
        layout.strides[dim] = layout.strides[dim - 1] * layout.padded_shape[dim - 1];
    }
    layout.total_size = static_cast<std::uint64_t>(layout.padded_shape[0]) *
                        layout.padded_shape[1] *
                        layout.padded_shape[2] *
                        layout.padded_shape[3];
    return layout;
}

}  // namespace gsc
