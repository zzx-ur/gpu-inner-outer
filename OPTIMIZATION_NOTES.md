# Abstraction Kernel Optimization: Mask + 4D Prefix Sum

## Overview

优化了`abstraction_kernel`中的约束检查逻辑，将其从逐个状态的for循环改为基于掩码和四维前缀和的快速查询方式。

## 原始实现的问题

在原始的`abstraction_kernel`中，对每个(state, input)对，需要检查后继盒内的所有状态是否满足约束：

```cpp
// 原始实现：for循环遍历所有后继状态
if (!d_box_satisfies_constraint(state_grid, box, constraint_params)) {
    // 内部实现：
    // for (w = box.min[3]; w <= box.max[3]; ++w) {
    //     for (z = box.min[2]; ; ++z) {
    //         for (y = box.min[1]; y <= box.max[1]; ++y) {
    //             for (x = box.min[0]; x <= box.max[0]; ++x) {
    //                 if (!d_satisfies_constraint(center, params)) {
    //                     return false;
    //                 }
    //             }
    //         }
    //     }
    // }
}
```

**性能问题：**
- 对于大的后继盒，需要遍历数千个状态
- 每个状态都需要计算中心坐标并检查约束
- 在GPU上，这导致大量的串行操作和分支

## 优化方案

### 1. 预构建约束掩码

在abstraction阶段开始前，为所有状态构建一个约束满足掩码：

```cpp
__global__ void build_constraint_mask_kernel(const Grid4D<std::uint32_t> state_grid,
                                              const GpuConstraintParams constraint_params,
                                              std::uint8_t* constraint_mask) {
    const std::uint64_t flat = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= state_grid.total_size) {
        return;
    }

    double center[kStateDim];
    state_grid.center(static_cast<std::uint32_t>(flat), center);
    constraint_mask[flat] = d_satisfies_constraint(center, constraint_params) ? 1 : 0;
}
```

**优势：**
- 所有状态的约束检查并行化
- 每个状态只检查一次
- 结果存储在掩码中供后续查询

### 2. 构建四维前缀和

对约束掩码和有效掩码分别构建四维前缀和：

```cpp
// 约束前缀和：prefix_constraint[i,j,k,l] = 
//   sum of constraint_mask[x,y,z,w] for x<=i, y<=j, z<=k, w<=l

// 有效掩码前缀和：prefix_valid[i,j,k,l] = 
//   sum of valid_mask[x,y,z,w] for x<=i, y<=j, z<=k, w<=l
```

使用现有的`build_prefix_on_device`函数，通过scatter + 4个scan kernels实现。

### 3. 快速约束检查

在abstraction_kernel中，使用前缀和快速查询替代for循环：

```cpp
__device__ bool d_box_satisfies_constraint_fast(const Grid4D<std::uint32_t>& grid,
                                                const PrefixLayout4D& layout,
                                                const PrefixCount* prefix_constraint,
                                                const PrefixCount* prefix_valid,
                                                const IndexBox4D& box) {
    // 查询盒内满足约束的状态数
    const PrefixCount constraint_count = d_query_box_count(grid, layout, prefix_constraint, 
                                                           grid.flatten(box.min), grid.flatten(box.max));
    
    // 查询盒内有效的状态数
    const PrefixCount valid_count = d_query_box_count(grid, layout, prefix_valid,
                                                      grid.flatten(box.min), grid.flatten(box.max));
    
    // 所有有效状态都满足约束当且仅当两个计数相等
    return constraint_count == valid_count;
}
```

**关键思想：**
- 使用16个顶点的容斥原理快速计算盒内元素和
- O(1)时间复杂度（相对于盒的大小）
- 避免了for循环中的分支和内存访问

## 实现细节

### 修改的函数

1. **`build_constraint_mask_kernel`** (新增)
   - 为每个状态计算约束满足情况
   - 并行度：state_count个线程

2. **`d_box_satisfies_constraint_fast`** (新增)
   - 使用前缀和快速检查盒内约束
   - 替代原来的`d_box_satisfies_constraint`

3. **`abstraction_kernel`** (修改)
   - 添加`PrefixLayout4D layout`参数
   - 添加`prefix_constraint`和`prefix_valid`参数
   - 调用`d_box_satisfies_constraint_fast`替代`d_box_satisfies_constraint`

4. **`run_case_cuda_u32`** (修改)
   - 添加`d_constraint_mask`显存分配
   - 添加`d_prefix_constraint`显存分配
   - 在abstraction阶段构建约束掩码和前缀和
   - 传递前缀和给abstraction_kernel

### 显存增加

- `d_constraint_mask`: state_count × 1字节
- `d_prefix_constraint`: prefix_layout.total_size × 8字节

总增加：约为state_count + 8×(n0+1)×(n1+1)×(n2+1)×(n3+1)字节

对于典型的204800状态的网格，增加约为：
- constraint_mask: 200KB
- prefix_constraint: 约2-4MB（取决于网格大小）

### 计算流程

```
1. build_constraint_mask_kernel
   ↓
2. build_prefix_on_device(d_constraint_mask → d_prefix_constraint)
   ├─ scatter_mask_kernel
   ├─ scan_axis0_kernel
   ├─ scan_axis1_kernel
   ├─ scan_axis2_kernel
   └─ scan_axis3_kernel
   ↓
3. build_prefix_on_device(d_valid_mask → d_prefix_valid)
   ├─ scatter_mask_kernel
   ├─ scan_axis0_kernel
   ├─ scan_axis1_kernel
   ├─ scan_axis2_kernel
   └─ scan_axis3_kernel
   ↓
4. abstraction_kernel (使用d_prefix_constraint和d_prefix_valid)
   ├─ 对每个(state, input)对
   ├─ 计算后继盒
   ├─ 调用d_box_satisfies_constraint_fast
   │  └─ 使用前缀和O(1)查询
   └─ 输出min_flat, max_flat, valid
```

## 性能分析

### 时间复杂度

**原始实现：**
- 对每个(state, input)对：O(box_volume)
- 总体：O(pair_count × avg_box_volume)
- 对于大盒子，可能是O(pair_count × state_count)

**优化实现：**
- 约束掩码构建：O(state_count)
- 前缀和构建：O(state_count)
- 对每个(state, input)对：O(1)（16个顶点查询）
- 总体：O(state_count + pair_count)

### 预期加速

对于后继盒平均大小为B的情况：
- 加速比 ≈ B（对于大盒子可能是10-100倍）

### 权衡

**优势：**
- 显著减少abstraction_kernel的执行时间
- 避免了for循环中的分支和内存访问
- 充分利用GPU的并行性

**成本：**
- 增加显存使用（约2-4MB）
- 增加abstraction阶段的前缀和构建时间
- 但总体时间仍然显著减少

## 验证

优化后的代码保持了原始的语义：
- 对于无约束情况（constraint_type == 0），两个实现都返回true
- 对于有约束情况，使用前缀和的计数方式等价于逐个检查

## 参考

这个优化方案参考了可达集合迭代中的前缀和查询方式：
- `pair_satisfaction_kernel`中使用`d_query_box_count`快速查询
- `reduce_inputs_kernel`中使用前缀和进行高效的集合操作

通过将相同的技术应用于约束检查，实现了abstraction阶段的显著加速。
