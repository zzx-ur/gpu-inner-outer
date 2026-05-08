#include "gsc/cuda_api.hpp"

#if !GSC_HAS_CUDA

namespace gsc {

bool cuda_backend_compiled() {
    return false;
}

GpuRunReport run_case_cuda_u32(const CaseConfig&) {
    GpuRunReport report;
    report.message = "CUDA backend was not built";
    return report;
}

GpuRunReport run_contracted_case_cuda_u32(const CaseConfig&) {
    GpuRunReport report;
    report.message = "CUDA backend was not built";
    return report;
}

}  // namespace gsc

#endif
