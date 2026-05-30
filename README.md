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
Note that the inner-outer set verification use 464.089 ms(this is already enough)

the total memory 3.85 GiB including lots of useless data, copy to cpu using 3154.28 ms for visualization. 

the useful data is controller using 15.3Mb. meaning that the total copy time is far less than 3154.28ms. 

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
abstraction   : 302.719 ms
solve         : 162.215 ms
memory total  : 3.85 GiB

[GPU Detailed Timing]
=== Abstraction Phase ===
  Kernel execution      : 302.13 ms
  H2D memcpy            : 417.273 ms
  Total abstraction     : 302.719 ms

=== Reachability Iteration Phase ===
  Prefix build (total)  : 2.11685 ms
  Pair satisfaction     : 158.054 ms
  Reduce inputs         : 1.78838 ms
  Iteration memcpy      : 0.197654 ms
  Total solve           : 162.215 ms

=== Result Copy Phase ===
  D2H memcpy            : 3154.28 ms

=== Summary ===
  Total kernel time     : 464.089 ms
  Total memcpy time     : 3571.75 ms
  Total compute time    : 4035.84 ms
  Total elapsed time    : 464.934 ms

=== GPU Memory Usage ===
  Abstraction data      : 3.36 GiB
  Prefix sum data       : 103.38 MiB
  Iteration data        : 49.75 MiB
  Total allocated       : 3.51 GiB
  Peak usage            : 4.40 GiB

=== Per-Iteration Breakdown (first 5 and last 5) ===
  Iter 1: 13.6951 ms total [prefix: 0.186744 ms, satisfaction: 13.219 ms, reduction: 0.289034 ms] -> 52481 new reachable, 123510 new certified
  Iter 2: 13.4331 ms total [prefix: 0.178956 ms, satisfaction: 13.0425 ms, reduction: 0.211376 ms] -> 57431 new reachable, 1097 new certified
  Iter 3: 13.4373 ms total [prefix: 0.173077 ms, satisfaction: 13.0563 ms, reduction: 0.207675 ms] -> 91549 new reachable, 1502 new certified
  Iter 4: 13.442 ms total [prefix: 0.178789 ms, satisfaction: 13.0624 ms, reduction: 0.200532 ms] -> 122322 new reachable, 1501 new certified
  Iter 5: 13.4261 ms total [prefix: 0.176208 ms, satisfaction: 13.0597 ms, reduction: 0.189876 ms] -> 129730 new reachable, 1459 new certified
  ...
  Iter 8: 13.4862 ms total [prefix: 0.177702 ms, satisfaction: 13.1612 ms, reduction: 0.147102 ms] -> 144731 new reachable, 5579 new certified
  Iter 9: 13.548 ms total [prefix: 0.176857 ms, satisfaction: 13.2434 ms, reduction: 0.127501 ms] -> 144977 new reachable, 7315 new certified
  Iter 10: 13.6024 ms total [prefix: 0.171175 ms, satisfaction: 13.3229 ms, reduction: 0.108103 ms] -> 140279 new reachable, 7976 new certified
  Iter 11: 13.6245 ms total [prefix: 0.176303 ms, satisfaction: 13.3598 ms, reduction: 0.088223 ms] -> 122784 new reachable, 3939 new certified
  Iter 12: 13.644 ms total [prefix: 0.172261 ms, satisfaction: 13.399 ms, reduction: 0.072498 ms] -> 54686 new reachable, 128 new certified
