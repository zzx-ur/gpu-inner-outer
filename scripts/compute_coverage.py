#!/usr/bin/env python3
"""
计算认证百分比和状态空间信息
"""

import h5py
import numpy as np
from pathlib import Path

def main():
    result_file = Path("results/hyperbolic_demo_gpu.h5")
    
    if not result_file.exists():
        print(f"错误: 文件不存在: {result_file}")
        return 1
    
    with h5py.File(result_file, "r") as f:
        # 读取状态空间信息
        state_shape = f["/state/shape"][:]
        state_lb = f["/state/lb"][:]
        state_ub = f["/state/ub"][:]
        state_eta = f["/state/eta"][:]
        
        # 读取最后一次迭代的统计
        iters = f["iterations"]
        total_certified = np.array(iters["total_certified"])
        total_reachable = np.array(iters["total_reachable"])
        
        # 计算总状态空间大小
        total_states = np.prod(state_shape)
        
        print("=" * 60)
        print("状态空间配置")
        print("=" * 60)
        print(f"状态空间维度 (shape): {state_shape}")
        print(f"下界 (lb): {state_lb}")
        print(f"上界 (ub): {state_ub}")
        print(f"网格步长 (eta): {state_eta}")
        print(f"总状态数: {total_states:,}")
        
        print("\n" + "=" * 60)
        print("收敛分析")
        print("=" * 60)
        
        final_certified = total_certified[-1]
        final_reachable = total_reachable[-1]
        
        certified_pct = (final_certified / total_states) * 100
        reachable_pct = (final_reachable / total_states) * 100
        
        print(f"最终认证状态数: {final_certified:,}")
        print(f"最终可达状态数: {final_reachable:,}")
        print(f"总状态空间大小: {total_states:,}")
        print(f"\n认证覆盖率: {certified_pct:.2f}%")
        print(f"可达覆盖率: {reachable_pct:.2f}%")
        
        # 读取摘要中的候选状态数
        if "summary/gpu" in f:
            summary = f["summary/gpu"]
            summary_certified = int(summary["certified_candidate_states"][()])
            print(f"\n摘要中的认证候选数: {summary_certified:,}")
            print(f"摘要中的收敛状态: {bool(summary['converged'][()])}")
            print(f"摘要消息: {summary['message'][()]}")
        
        print("\n" + "=" * 60)
        print("迭代停止分析")
        print("=" * 60)
        print(f"总迭代次数: {len(total_certified)}")
        print(f"最后5次迭代新增认证数:")
        for i in range(max(0, len(total_certified)-5), len(total_certified)):
            print(f"  迭代 {i+1}: 新增认证 = {int(total_certified[i] - (total_certified[i-1] if i > 0 else 0)):,}, "
                  f"累计认证 = {int(total_certified[i]):,}")
        
        # 判断收敛情况
        if total_certified[-1] == total_certified[-2]:
            print(f"\n结论: 迭代在第 {len(total_certified)} 次停滞（新增认证为0）")
            print(f"认证进度: {certified_pct:.2f}% ({final_certified:,}/{total_states:,})")

if __name__ == "__main__":
    exit(main())
