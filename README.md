# CUDA Symbolic Control

C++17/CUDA project implementing grid symbolic control for K-invariance verification.

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

python3 scripts/extract_iterations.py

文件: results/hyperbolic_demo_gpu.h5
总迭代次数: 19
显示前 19 次迭代

------------------------------------------------------------------------------------------------------------------
    迭代 |       新增可达 |       新增认证 |       累计可达 |       累计认证 |       迭代耗时(ms) |       前缀构建 |       满足检查 |       输入归约
------------------------------------------------------------------------------------------------------------------
     1 |      68114 |     152653 |     269714 |     152653 |         11.610 |      0.206 |     10.988 |      0.416
     2 |      56917 |       1197 |     326631 |     153850 |         11.472 |      0.198 |     10.919 |      0.354
     3 |      79383 |       1511 |     406014 |     155361 |         11.426 |      0.197 |     10.904 |      0.324
     4 |      88525 |       1770 |     494539 |     157131 |         11.399 |      0.197 |     10.908 |      0.293
     5 |      76583 |       1158 |     571122 |     158289 |         11.380 |      0.197 |     10.916 |      0.267
     6 |      67571 |       1495 |     638693 |     159784 |         11.364 |      0.197 |     10.913 |      0.254
     7 |      59356 |       1804 |     698049 |     161588 |         11.357 |      0.197 |     10.919 |      0.241
     8 |      46879 |       3606 |     744928 |     165194 |         11.329 |      0.197 |     10.905 |      0.227
     9 |      33909 |       4597 |     778837 |     169791 |         11.333 |      0.195 |     10.914 |      0.224
    10 |      22599 |       5710 |     801436 |     175501 |         11.333 |      0.197 |     10.921 |      0.215
    11 |      13974 |       6051 |     815410 |     181552 |         11.317 |      0.198 |     10.916 |      0.203
    12 |       8481 |       5428 |     823891 |     186980 |         11.312 |      0.196 |     10.918 |      0.198
    13 |       4687 |       4721 |     828578 |     191701 |         11.308 |      0.196 |     10.918 |      0.194
    14 |       1954 |       2078 |     830532 |     193779 |         11.314 |      0.196 |     10.926 |      0.191
    15 |        404 |        342 |     830936 |     194121 |         11.308 |      0.197 |     10.921 |      0.190
    16 |        162 |        109 |     831098 |     194230 |         11.295 |      0.196 |     10.908 |      0.191
    17 |         57 |         31 |     831155 |     194261 |         11.302 |      0.196 |     10.914 |      0.192
    18 |          6 |         10 |     831161 |     194271 |         11.301 |      0.195 |     10.913 |      0.192
    19 |          0 |          0 |     831161 |     194271 |         11.302 |      0.195 |     10.911 |      0.195
------------------------------------------------------------------------------------------------------------------

统计信息:
  平均迭代耗时: 11.36 ms
  总认证状态数: 194271
  总可达状态数: 831161

GPU求解摘要:
  收敛状态: False
  消息: b'iteration stalled before candidate closure'
  抽象阶段: 556.65 ms
  求解阶段: 215.87 ms
