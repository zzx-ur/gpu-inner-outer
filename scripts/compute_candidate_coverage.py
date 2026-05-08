#!/usr/bin/env python3
"""
计算相对于候选集的认证百分比
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
        state_eta = f["/state/eta"][:]
        
        # 读取候选集边界
        candidate_lb = f["/config/candidate_lb"][:]
        candidate_ub = f["/config/candidate_ub"][:]
        
        # 读取最后一次迭代的统计
        iters = f["iterations"]
        total_certified = np.array(iters["total_certified"])
        
        # 计算总状态空间大小
        total_states = np.prod(state_shape)
        
        # 计算候选集状态数
        candidate_cells = []
        for dim in range(4):
            # 找到候选边界对应的网格索引范围
            lb_val = candidate_lb[dim]
            ub_val = candidate_ub[dim]
            eta = state_eta[dim]
            
            # 网格边界从 state_lb[dim] 开始
            grid_lb = state_lb[dim]
            
            # 计算索引范围
            idx_start = int(np.ceil((lb_val - grid_lb) / eta))
            idx_end = int(np.floor((ub_val - grid_lb) / eta))
            
            # 确保在有效范围内
            idx_start = max(0, idx_start)
            idx_end = min(state_shape[dim] - 1, idx_end)
            
            n_cells = idx_end - idx_start + 1
            candidate_cells.append(n_cells)
        
        candidate_total = np.prod(candidate_cells)
        
        print("=" * 60)
        print("候选集 (candidate set) 分析")
        print("=" * 60)
        print(f"候选集边界:")
        print(f"  下界: {candidate_lb}")
        print(f"  上界: {candidate_ub}")
        print(f"\n状态网格信息:")
        print(f"  总网格形状: {state_shape}")
        print(f"  网格下界: {state_lb}")
        print(f"  网格步长: {state_eta}")
        print(f"\n候选集在各维度的网格数: {candidate_cells}")
        print(f"候选集总状态数: {candidate_total:,}")
        
        print("\n" + "=" * 60)
        print("收敛进度")
        print("=" * 60)
        
        final_certified = int(total_certified[-1])
        
        certified_pct = (final_certified / candidate_total) * 100
        
        print(f"最终认证候选状态数: {final_certified:,}")
        print(f"候选集总状态数: {candidate_total:,}")
        print(f"\n认证覆盖率: {certified_pct:.2f}%")
        print(f"未认证数: {candidate_total - final_certified:,}")
        
        # 检查摘要
        if "summary/gpu" in f:
            summary = f["summary/gpu"]
            summary_certified = int(summary["certified_candidate_states"][()])
            print(f"\n摘要中记录的认证数: {summary_certified:,}")
            print(f"与迭代数据一致: {summary_certified == final_certified}")
        
        # 判断是否收敛
        print(f"\n收敛状态: {'已收敛' if certified_pct >= 100.0 else '未收敛'}")
        if certified_pct < 100:
            print(f"  在第 {len(total_certified)} 次迭代时停止")
            print(f"  最后5次迭代的新增认证:")
            for i in range(max(0, len(total_certified)-5), len(total_certified)):
                prev_cert = total_certified[i-1] if i > 0 else 0
                newly = int(total_certified[i] - prev_cert)
                print(f"    迭代 {i+1}: +{newly:,} (累计 {int(total_certified[i]):,})")

if __name__ == "__main__":
    exit(main())
