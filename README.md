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

# Run CPU mode
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg --cpu

# Run GPU mode (standard backward reachability)
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg --gpu

# Run GPU mode (contracted hyperbolic constraint forward reachability)
# Uses contracted candidate set and contracted constraints to achieve >=95% coverage of original candidate
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg --gpu-contracted
```

## Modes

| Mode | Description |
|------|-------------|
| `--cpu` | Standard backward reachability on CPU |
| `--gpu` | GPU-accelerated backward reachability |
| `--gpu-contracted` | **New**: Forward reachability using contracted hyperbolic constraints. Requires `candidate_contracted` and `disturbance_half_width = 0,0,0,0` in config |

## Contracted Constraint Mode

The `--gpu-contracted` mode implements forward reachability analysis using:
- A contracted constraint region (smaller than original)
- A contracted candidate set (subset of original candidate)
- Nominal dynamics (zero disturbance)

**Configuration requirements:**
```ini
# Required: Contracted candidate set
candidate_contracted_lb = -0.8, -0.8, -3.141592653589793, 0.0
candidate_contracted_ub =  0.8,  0.8,  3.141592653589793, 1.0

# Required: Contracted constraint parameters
hyperbolic_contracted_a = 2.56   # = 4.0 * 0.8^2
hyperbolic_contracted_b = 4.0
hyperbolic_contracted_c = 10.24  # = 16.0 * 0.8^2

# Required: Zero disturbance for nominal model
disturbance_half_width = 0.0, 0.0, 0.0, 0.0
```

**Termination:** Iterations stop when coverage of original candidate reaches ≥95%.

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


# 完全重建
# 删除现有构建目录
rm -rf build

# 重新配置（启用 CUDA）
cmake -S . -B build -DGSC_ENABLE_CUDA=ON

# 或 CPU -only 构建
 cmake -S . -B build -DGSC_ENABLE_CUDA=OFF

# 编译（使用所有可用核心）
cmake --build build -j $(sysctl -n hw.ncpu)
# GPU 模式
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg --gpu

# 收缩约束前向可达性
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg --gpu-contracted


# results

GPU: Abstraction phase completed in 566.961 ms
GPU: Starting forward reachability iterations...
GPU: Iteration 1/100, newly reachable: 42196, newly certified: 106253, total certified: 106253/161280 (coverage: 65.9%) (17.8 ms)
GPU: Iteration 2/100, newly reachable: 45437, newly certified: 4213, total certified: 110466/161280 (coverage: 68.5%) (17.6 ms)
GPU: Iteration 3/100, newly reachable: 67878, newly certified: 2985, total certified: 113451/161280 (coverage: 70.3%) (17.6 ms)
GPU: Iteration 4/100, newly reachable: 71893, newly certified: 2470, total certified: 115921/161280 (coverage: 71.9%) (17.6 ms)
GPU: Iteration 5/100, newly reachable: 64957, newly certified: 3562, total certified: 119483/161280 (coverage: 74.1%) (17.6 ms)
GPU: Iteration 10/100, newly reachable: 25085, newly certified: 9836, total certified: 157055/161280 (coverage: 97.4%) (17.6 ms)
GPU: Reached target coverage of 97.4%
GPU: Solve phase completed in 176.2 ms
GPU: Final status - converged: true, iterations: 10, message: candidate coverage >= 95%
GPU: Copying abstraction data to host...

[gpu-contracted]
states        : 4013100
inputs        : 100
pairs         : 401310000
reachable     : 592398
candidate     : 161280
candidate ctl : 157055
iterations    : 10
converged     : true
message       : candidate coverage >= 95%
abstraction   : 567.0 ms
solve         : 176.2 ms
memory total  : 3.85 GiB