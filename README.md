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

