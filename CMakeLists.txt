cmake_minimum_required(VERSION 3.22.1)

project(cuda_symbolic_control LANGUAGES CXX C)

include(CheckLanguage)
include(CTest)

option(GSC_ENABLE_CUDA "Build the CUDA backend when a CUDA toolkit is available." ON)
option(GSC_ENABLE_HDF5_EXPORT "Build HDF5 result export for CLI runs." ON)

add_library(gsc_core STATIC
    src/config.cpp
    src/cuda_stubs.cpp
    src/result_io.cpp
)

target_include_directories(gsc_core PUBLIC include)
target_compile_features(gsc_core PUBLIC cxx_std_17)
target_compile_options(gsc_core PRIVATE -Wall -Wextra -Wpedantic)

set(GSC_HAS_CUDA 0)
if(GSC_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        add_library(gsc_cuda STATIC cuda/kernels.cu)
        target_include_directories(gsc_cuda PUBLIC include)
        target_compile_features(gsc_cuda PUBLIC cxx_std_17)
        set_target_properties(gsc_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_link_libraries(gsc_cuda PUBLIC gsc_core CUDA::cudart)
        set(GSC_HAS_CUDA 1)
    endif()
endif()

target_compile_definitions(gsc_core PUBLIC GSC_HAS_CUDA=${GSC_HAS_CUDA})

set(GSC_HAS_HDF5 0)
if(GSC_ENABLE_HDF5_EXPORT)
    find_package(HDF5 QUIET COMPONENTS C)
    if(HDF5_FOUND)
        target_link_libraries(gsc_core PUBLIC HDF5::HDF5)
        set(GSC_HAS_HDF5 1)
    else()
        message(WARNING "HDF5 was not found; result export support will be disabled.")
    endif()
endif()

target_compile_definitions(gsc_core PUBLIC GSC_HAS_HDF5=${GSC_HAS_HDF5})

add_executable(cuda_symbolic_control_cli src/main.cpp)
target_link_libraries(cuda_symbolic_control_cli PRIVATE gsc_core)
if(TARGET gsc_cuda)
    target_link_libraries(cuda_symbolic_control_cli PRIVATE gsc_cuda)
endif()

if(BUILD_TESTING)
    add_executable(cuda_symbolic_control_test_small tests/test_small_case.cpp)
    target_link_libraries(cuda_symbolic_control_test_small PRIVATE gsc_core)
    if(TARGET gsc_cuda)
        target_link_libraries(cuda_symbolic_control_test_small PRIVATE gsc_cuda)
    endif()
    add_test(NAME cuda_symbolic_control_test_small COMMAND cuda_symbolic_control_test_small)

    add_executable(cuda_symbolic_control_smoke_test tests/smoke_test.cpp)
    target_link_libraries(cuda_symbolic_control_smoke_test PRIVATE gsc_core)
    if(TARGET gsc_cuda)
        target_link_libraries(cuda_symbolic_control_smoke_test PRIVATE gsc_cuda)
    endif()
    add_test(NAME cuda_symbolic_control_smoke_test COMMAND cuda_symbolic_control_smoke_test)

    add_executable(cuda_symbolic_control_result_io_test tests/test_result_io.cpp)
    target_link_libraries(cuda_symbolic_control_result_io_test PRIVATE gsc_core)
    if(TARGET gsc_cuda)
        target_link_libraries(cuda_symbolic_control_result_io_test PRIVATE gsc_cuda)
    endif()
    add_test(NAME cuda_symbolic_control_result_io_test COMMAND cuda_symbolic_control_result_io_test)
endif()
