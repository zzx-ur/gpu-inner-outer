# CUDA Symbolic Control

C++17/CUDA project implementing grid symbolic control for K-invariance verification.
hardware : RTX 4090 24G

## Quick Start

```bash
# CPU-only build
cmake -S . -B build_cpu -DGSC_ENABLE_CUDA=OFF
cmake --build build_cpu -j 4

# With CUDA
cmake -S . -B build
cmake --build build -j 8

# Run
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg --gpu
```

## Documentation

- **ALGORITHM.md** — Dynamics abstraction, candidate/reachable sets, backward reachability
- **CUDA_ACCELERATION.md** — GPU kernels, optimization strategies, performance data
- **VISUALIZATION.md** — Plotting scripts for HDF5 results

## Directory Structure

```
├── ALGORITHM.md           # Algorithm documentation
├── CUDA_ACCELERATION.md   # GPU documentation
├── VISUALIZATION.md       # Visualization guide
├── include/gsc/           # Core headers
├── src/                   # Main implementation
├── cuda/                  # CUDA kernels
├── scripts/               # Python visualization tools
├── examples/              # Configuration files
└── legacy/                # Archived MATLAB code
```

## Potential problem of the existing algorithm

The current algorithm may have some boundary issues; generally, if the coverage of candidate kinv exceeds 90%, it is considered viable.

## Results

GPU: Starting abstraction phase with 308700000 state-input pairs...
GPU: Abstraction phase completed in 783.722 ms (kernel: 783.652 ms)
GPU: Iteration 1/100, newly reachable: 37737, newly certified: 210434, total certified: 210434/252000 (12.9378 ms)
GPU: Iteration 2/100, newly reachable: 67308, newly certified: 1424, total certified: 211858/252000 (12.7981 ms)
GPU: Iteration 3/100, newly reachable: 94968, newly certified: 2730, total certified: 214588/252000 (12.7799 ms)
GPU: Iteration 4/100, newly reachable: 112363, newly certified: 2785, total certified: 217373/252000 (12.7452 ms)
GPU: Iteration 5/100, newly reachable: 99473, newly certified: 3232, total certified: 220605/252000 (12.5986 ms)
GPU: Iteration 10/100, newly reachable: 34319, newly certified: 4014, total certified: 251660/252000 (12.4909 ms)
GPU: Solve phase completed in 151.604 ms
GPU: Final status - converged: true, iterations: 12, message: all candidate states obtained a controller
GPU: Copying abstraction data to host (2649.59 MB)...

[gpu]
states        : 3087000
inputs        : 100
pairs         : 308700000
reachable     : 1004958
candidate     : 252000
candidate ctl : 252000
iterations    : 12
converged     : true
message       : all candidate states obtained a controller
abstraction   : 783.722 ms
solve         : 151.604 ms
memory total  : 2.96 GiB
result file   : results/hyperbolic_demo_gpu.h5
  - includes abstraction data (2649.59 MB)
  - includes 12 iteration statistics
