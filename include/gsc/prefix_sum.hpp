#pragma once

#include <vector>

#include "gsc/grid.hpp"

namespace gsc {

template <typename StateIndex>
struct PrefixGrid4D {
    PrefixLayout4D layout{};
    std::vector<PrefixCount> data;
};

template <typename StateIndex>
inline PrefixGrid4D<StateIndex> build_prefix_sum_cpu(const Grid4D<StateIndex>& grid,
                                                     const std::vector<std::uint8_t>& mask) {
    PrefixGrid4D<StateIndex> prefix;
    prefix.layout = build_prefix_layout(grid);
    prefix.data.assign(prefix.layout.total_size, 0);

    for (std::uint64_t w = 1; w < prefix.layout.padded_shape[3]; ++w) {
        for (std::uint64_t z = 1; z < prefix.layout.padded_shape[2]; ++z) {
            for (std::uint64_t y = 1; y < prefix.layout.padded_shape[1]; ++y) {
                for (std::uint64_t x = 1; x < prefix.layout.padded_shape[0]; ++x) {
                    const std::uint32_t idx[4] = {
                        static_cast<std::uint32_t>(x - 1),
                        static_cast<std::uint32_t>(y - 1),
                        static_cast<std::uint32_t>(z - 1),
                        static_cast<std::uint32_t>(w - 1),
                    };
                    const auto flat = static_cast<std::uint64_t>(grid.flatten(idx));
                    const PrefixCount value = mask[flat] ? 1 : 0;

                    const auto p = prefix.layout.offset(static_cast<std::uint32_t>(x),
                                                        static_cast<std::uint32_t>(y),
                                                        static_cast<std::uint32_t>(z),
                                                        static_cast<std::uint32_t>(w));

                    prefix.data[p] =
                        value
                        + prefix.data[prefix.layout.offset(x - 1, y, z, w)]
                        + prefix.data[prefix.layout.offset(x, y - 1, z, w)]
                        + prefix.data[prefix.layout.offset(x, y, z - 1, w)]
                        + prefix.data[prefix.layout.offset(x, y, z, w - 1)]
                        - prefix.data[prefix.layout.offset(x - 1, y - 1, z, w)]
                        - prefix.data[prefix.layout.offset(x - 1, y, z - 1, w)]
                        - prefix.data[prefix.layout.offset(x - 1, y, z, w - 1)]
                        - prefix.data[prefix.layout.offset(x, y - 1, z - 1, w)]
                        - prefix.data[prefix.layout.offset(x, y - 1, z, w - 1)]
                        - prefix.data[prefix.layout.offset(x, y, z - 1, w - 1)]
                        + prefix.data[prefix.layout.offset(x - 1, y - 1, z - 1, w)]
                        + prefix.data[prefix.layout.offset(x - 1, y - 1, z, w - 1)]
                        + prefix.data[prefix.layout.offset(x - 1, y, z - 1, w - 1)]
                        + prefix.data[prefix.layout.offset(x, y - 1, z - 1, w - 1)]
                        - prefix.data[prefix.layout.offset(x - 1, y - 1, z - 1, w - 1)];
                }
            }
        }
    }

    return prefix;
}

inline PrefixCount query_box_single(const PrefixLayout4D& layout,
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
        if ((__builtin_popcount(static_cast<unsigned>(mask)) & 1) == 0) {
            total += value;
        } else {
            total -= value;
        }
    }
    return total;
}

template <typename StateIndex>
inline PrefixCount query_box_count(const Grid4D<StateIndex>& grid,
                                   const PrefixGrid4D<StateIndex>& prefix,
                                   StateIndex min_flat,
                                   StateIndex max_flat) {
    IndexBox4D box{};
    index_box_from_flats(grid, min_flat, max_flat, &box);

    if (box.min[grid.wrap_dim] <= box.max[grid.wrap_dim]) {
        return query_box_single(prefix.layout, prefix.data.data(), box);
    }

    IndexBox4D left = box;
    IndexBox4D right = box;
    left.max[grid.wrap_dim] = grid.shape[grid.wrap_dim] - 1;
    right.min[grid.wrap_dim] = 0;
    return query_box_single(prefix.layout, prefix.data.data(), left)
         + query_box_single(prefix.layout, prefix.data.data(), right);
}

}  // namespace gsc
