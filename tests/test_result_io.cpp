#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "gsc/result_io.hpp"

#if GSC_HAS_HDF5
#include <hdf5.h>
#endif

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        const auto cfg = gsc::make_default_case();
        
        // GPU-only computation
        const auto gpu = gsc::run_case_cuda_u32(cfg);
        expect(gpu.executed, "gpu backend run did not execute");

        const auto output_path = std::filesystem::temp_directory_path() / "gsc_result_io_test.h5";
        std::filesystem::remove(output_path);

        gsc::write_gpu_results_hdf5(output_path.string(), cfg, gpu);

        expect(std::filesystem::exists(output_path), "expected HDF5 output file to exist");
        expect(std::filesystem::file_size(output_path) > 0, "expected HDF5 output file to be non-empty");

#if GSC_HAS_HDF5
        const hid_t file = H5Fopen(output_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        expect(file >= 0, "expected HDF5 file to open");
        expect(H5Lexists(file, "/result/gpu/controller", H5P_DEFAULT) > 0, "missing controller dataset");
        expect(H5Lexists(file, "/summary/gpu/message", H5P_DEFAULT) > 0, "missing summary message dataset");
        H5Fclose(file);
#endif

        std::filesystem::remove(output_path);
        std::cout << "result I/O export checks passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "test_result_io failed: " << ex.what() << '\n';
        return 1;
    }
}
