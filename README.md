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


## Results

[gpu]
states        : 4013100
inputs        : 100
pairs         : 401310000
reachable     : 1142689
candidate     : 161280
candidate ctl : 161280
iterations    : 12
converged     : true
message       : all candidate states obtained a controller
abstraction   : 1002.53 ms
solve         : 214.936 ms
memory total  : 3.85 GiB

[GPU Detailed Timing]
=== Abstraction Phase ===
  Kernel execution      : 1002.46 ms
  H2D memcpy            : 41.545 ms
  Total abstraction     : 1002.53 ms

=== Reachability Iteration Phase ===
  Prefix build (total)  : 3.41363 ms
  Pair satisfaction     : 207.635 ms
  Reduce inputs         : 3.55184 ms
  Iteration memcpy      : 0.21035 ms
  Total solve           : 214.936 ms

=== Result Copy Phase ===
  D2H memcpy            : 2869.32 ms

=== Summary ===
  Total kernel time     : 1217.06 ms
  Total memcpy time     : 2911.08 ms
  Total compute time    : 4128.14 ms
  Total elapsed time    : 1217.47 ms

=== GPU Memory Usage ===
  Abstraction data      : 3.36 GiB
  Prefix sum data       : 68.92 MiB
  Iteration data        : 45.93 MiB
  Total allocated       : 3.48 GiB
  Peak usage            : 4.25 GiB

=== Per-Iteration Breakdown (first 5 and last 5) ===
  Iter 1: 18.0966 ms total [prefix: 0.293492 ms, satisfaction: 17.3113 ms, reduction: 0.491322 ms] -> 52481 new reachable, 123510 new certified
  Iter 2: 18.0345 ms total [prefix: 0.28482 ms, satisfaction: 17.3104 ms, reduction: 0.438891 ms] -> 57431 new reachable, 1097 new certified
  Iter 3: 17.9946 ms total [prefix: 0.283121 ms, satisfaction: 17.2934 ms, reduction: 0.41772 ms] -> 91549 new reachable, 1502 new certified
  Iter 4: 17.9563 ms total [prefix: 0.283601 ms, satisfaction: 17.2824 ms, reduction: 0.38993 ms] -> 122322 new reachable, 1501 new certified
  Iter 5: 17.915 ms total [prefix: 0.28432 ms, satisfaction: 17.2794 ms, reduction: 0.350821 ms] -> 129220 new reachable, 1459 new certified
  ...
  Iter 8: 17.8205 ms total [prefix: 0.28284 ms, satisfaction: 17.2838 ms, reduction: 0.25349 ms] -> 94948 new reachable, 5579 new certified
  Iter 9: 17.7998 ms total [prefix: 0.28278 ms, satisfaction: 17.2833 ms, reduction: 0.23329 ms] -> 78464 new reachable, 7315 new certified
  Iter 10: 17.7951 ms total [prefix: 0.28432 ms, satisfaction: 17.2969 ms, reduction: 0.21347 ms] -> 64643 new reachable, 7976 new certified
  Iter 11: 17.8138 ms total [prefix: 0.282501 ms, satisfaction: 17.326 ms, reduction: 0.204851 ms] -> 43185 new reachable, 3937 new certified
  Iter 12: 17.8678 ms total [prefix: 0.283761 ms, satisfaction: 17.3849 ms, reduction: 0.198609 ms] -> 16768 new reachable, 130 new certified

result file   : results/hyperbolic_demo_gpu.h5
  - includes abstraction data (3444.47 MB)
  - includes 12 iteration statistics
