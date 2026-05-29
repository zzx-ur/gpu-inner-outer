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
reachable     : 1487115
candidate     : 161280
candidate ctl : 161280
iterations    : 12
converged     : true
message       : all candidate states obtained a controller
abstraction   : 659.878 ms
solve         : 161.927 ms
memory total  : 3.85 GiB

[GPU Detailed Timing]
=== Abstraction Phase ===
  Kernel execution      : 303.513 ms
  H2D memcpy            : 33.9179 ms
  Total abstraction     : 659.878 ms

=== Reachability Iteration Phase ===
  Prefix build (total)  : 2.12908 ms
  Pair satisfaction     : 157.749 ms
  Reduce inputs         : 1.79294 ms
  Iteration memcpy      : 0.202461 ms
  Total solve           : 161.927 ms

=== Result Copy Phase ===
  D2H memcpy            : 2518.45 ms

=== Summary ===
  Total kernel time     : 465.184 ms
  Total memcpy time     : 2552.57 ms
  Total compute time    : 3017.75 ms
  Total elapsed time    : 821.805 ms

=== GPU Memory Usage ===
  Abstraction data      : 3.36 GiB
  Prefix sum data       : 103.38 MiB
  Iteration data        : 49.75 MiB
  Total allocated       : 3.51 GiB
  Peak usage            : 4.40 GiB

=== Per-Iteration Breakdown (first 5 and last 5) ===
  Iter 1: 13.7472 ms total [prefix: 0.186085 ms, satisfaction: 13.2635 ms, reduction: 0.297175 ms] -> 52481 new reachable, 123510 new certified
  Iter 2: 13.4816 ms total [prefix: 0.176967 ms, satisfaction: 13.0918 ms, reduction: 0.212544 ms] -> 57431 new reachable, 1097 new certified
  Iter 3: 13.4655 ms total [prefix: 0.177258 ms, satisfaction: 13.0794 ms, reduction: 0.208562 ms] -> 91549 new reachable, 1502 new certified
  Iter 4: 13.4447 ms total [prefix: 0.176069 ms, satisfaction: 13.0673 ms, reduction: 0.201031 ms] -> 122322 new reachable, 1501 new certified
  Iter 5: 13.4387 ms total [prefix: 0.181446 ms, satisfaction: 13.0658 ms, reduction: 0.191201 ms] -> 129730 new reachable, 1459 new certified
  ...
  Iter 8: 13.4589 ms total [prefix: 0.175872 ms, satisfaction: 13.1343 ms, reduction: 0.148526 ms] -> 144731 new reachable, 5579 new certified
  Iter 9: 13.4796 ms total [prefix: 0.175129 ms, satisfaction: 13.1767 ms, reduction: 0.127445 ms] -> 144977 new reachable, 7315 new certified
  Iter 10: 13.4876 ms total [prefix: 0.176515 ms, satisfaction: 13.2029 ms, reduction: 0.107891 ms] -> 140279 new reachable, 7976 new certified
  Iter 11: 13.4754 ms total [prefix: 0.17393 ms, satisfaction: 13.2126 ms, reduction: 0.088614 ms] -> 122784 new reachable, 3939 new certified
  Iter 12: 13.4983 ms total [prefix: 0.176201 ms, satisfaction: 13.2503 ms, reduction: 0.071592 ms] -> 54686 new reachable, 128 new certified