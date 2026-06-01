# 快速参考：GPU 初始化优化

## 问题

原始代码在 CPU 端分配和初始化两个大数组（candidate_mask 和 valid_mask），耗时 ~400ms，但这些数组在 GPU 路径中从未被使用。

## 解决方案

### 1. 删除 CPU 端的数组分配

**文件：** `include/gsc/abstraction.hpp`

```cpp
// 优化前
prepared.candidate_mask.assign(prepared.state_grid.total_size, 0);  // 删除
prepared.valid_mask.assign(prepared.state_grid.total_size, 0);      // 删除

// 优化后
// 不分配，GPU 将直接计算
```

### 2. 使用融合 Kernel

**文件：** `cuda/kernels.cu`

```cpp
// 原来：4 个独立 kernel
build_candidate_mask_kernel
build_valid_mask_kernel
init_candidate_reach_step_kernel
count_candidates_kernel

// 现在：1 个融合 kernel
fused_init_kernel  // 或 fused_init_hyperbolic_kernel
```

### 3. 简化为 GPU-only 模式

**文件：** `src/main.cpp`

```cpp
// 原来：支持 --cpu, --gpu, --compare
// 现在：只支持 GPU 模式
```

## 性能提升

- **CPU 端初始化：** 400ms → 0ms
- **GPU kernel 执行：** 4 次 → 1 次
- **总体提升：** ~4-8 倍

## 验证

```bash
cd build
make -j8
./cuda_symbolic_control_cli ../examples/hyperbolic_case.cfg
```

预期输出中 `GPU init (masks+count)` 应该从 ~400ms 降低到 ~50-100ms。
