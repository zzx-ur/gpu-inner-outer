#pragma once

#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define GSC_HD __host__ __device__
#else
#define GSC_HD
#endif

namespace gsc {

constexpr int kStateDim = 4;
constexpr int kInputDim = 2;

using PairIndex = std::uint64_t;
using PrefixCount = std::uint64_t;
using InputIndex = std::uint32_t;
using ReachStep = std::uint32_t;

constexpr InputIndex kInvalidInput = std::numeric_limits<InputIndex>::max();
constexpr ReachStep kUnreachableStep = std::numeric_limits<ReachStep>::max();

template <typename T>
constexpr bool requires_u64_index(std::uint64_t count) {
    return count > static_cast<std::uint64_t>(std::numeric_limits<T>::max());
}

}  // namespace gsc
