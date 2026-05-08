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

The current algorithm may have some boundary issues; generally, if the coverage of candidate kinv exceeds 99%, it is considered viable.

## 个人应用远程gpu计算
# 删除旧 build（可选）
rm -rf build

# 配置 CUDA 构建
cmake -S . -B build -DGSC_ENABLE_CUDA=ON

# 编译
cmake --build build -j 8

# 运行示例
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg --gpu

# 看结果
<!-- 
ls -la results/

rm -rf results/*
git pull

看文件：

git diff HEAD~1

看某个文件：

git diff HEAD~1 README.md -->