# CUDA Symbolic Control

C++17/CUDA project implementing grid symbolic control for K-invariance verification.
hardware : RTX 5090 D 32G
24 vCPU Intel(R) Xeon(R) Platinum 8470Q

## Quick Start

```bash
# CPU-only build
cmake -S . -B build_cpu -DGSC_ENABLE_CUDA=OFF
cmake --build build_cpu -j 4

# With CUDA
cmake -S . -B build
cmake --build build -j 8

# Run
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg 
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
Note that the inner-outer set verification use 464.089 ms(this is already enough)

the total memory 3.85 GiB including lots of useless data, copy to cpu using 3154.28 ms for visualization. 

the useful data is controller using 15.3Mb. meaning that the total copy time is far less than 3154.28ms. 

[GPU Detailed Timing]
=== Abstraction Phase ===
  Kernel execution      : 295.44 ms
  GPU init (masks+count): 456.922 ms
  Total abstraction     : 752.935 ms

=== Reachability Iteration Phase ===
  Prefix build (total)  : 2.15574 ms
  Pair satisfaction     : 151.56 ms
  Reduce inputs         : 1.83759 ms
  Iteration memcpy      : 0.180344 ms
  Total solve           : 155.819 ms

=== Result Copy Phase ===
  D2H memcpy            : 4699.66 ms

=== Summary ===
  Total kernel time     : 450.994 ms
  Total memcpy time     : 5156.76 ms
  Total compute time    : 5607.76 ms
  Total elapsed time    : 908.753 ms

=== GPU Memory Usage ===
  Abstraction data      : 3.36 GiB
  Prefix sum data       : 103.38 MiB
  Iteration data        : 49.75 MiB
  Total allocated       : 3.51 GiB
  Peak usage            : 4.39 GiB

=== Per-Iteration Breakdown (first 5 and last 5) ===
  Iter 1: 13.1312 ms total [prefix: 0.182252 ms, satisfaction: 12.704 ms, reduction: 0.244654 ms] -> 52481 new reachable, 123510 new certified
  Iter 2: 13.0081 ms total [prefix: 0.17967 ms, satisfaction: 12.6139 ms, reduction: 0.214323 ms] -> 57431 new reachable, 1097 new certified
  Iter 3: 12.9968 ms total [prefix: 0.177524 ms, satisfaction: 12.6098 ms, reduction: 0.20928 ms] -> 91549 new reachable, 1502 new certified
  Iter 4: 12.9834 ms total [prefix: 0.177131 ms, satisfaction: 12.6011 ms, reduction: 0.204947 ms] -> 122322 new reachable, 1501 new certified
  Iter 5: 13.0044 ms total [prefix: 0.185322 ms, satisfaction: 12.627 ms, reduction: 0.191833 ms] -> 129730 new reachable, 1459 new certified
  ...
  Iter 8: 12.9584 ms total [prefix: 0.178506 ms, satisfaction: 12.6304 ms, reduction: 0.149283 ms] -> 144731 new reachable, 5579 new certified
  Iter 9: 12.9458 ms total [prefix: 0.179855 ms, satisfaction: 12.6337 ms, reduction: 0.131999 ms] -> 144977 new reachable, 7315 new certified
  Iter 10: 12.9168 ms total [prefix: 0.175607 ms, satisfaction: 12.6298 ms, reduction: 0.111225 ms] -> 140279 new reachable, 7976 new certified
  Iter 11: 12.8985 ms total [prefix: 0.180316 ms, satisfaction: 12.6268 ms, reduction: 0.091101 ms] -> 122784 new reachable, 3939 new certified
  Iter 12: 12.9298 ms total [prefix: 0.184381 ms, satisfaction: 12.6236 ms, reduction: 0.12129 ms] -> 54686 new reachable, 128 new certified