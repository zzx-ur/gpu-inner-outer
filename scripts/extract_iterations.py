#!/usr/bin/env python3
"""
提取结果文件前20次迭代的信息
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
        # 检查是否有迭代数据
        if "iterations" not in f:
            print("错误: HDF5文件中没有找到 'iterations' 组")
            return 1
        
        iters = f["iterations"]
        
        # 读取所有迭代数据
        data = {
            "iteration": np.array(iters["iteration"]),
            "newly_reachable": np.array(iters["newly_reachable"]),
            "newly_certified": np.array(iters["newly_certified"]),
            "total_reachable": np.array(iters["total_reachable"]),
            "total_certified": np.array(iters["total_certified"]),
            "iteration_ms": np.array(iters["iteration_ms"]),
            "prefix_build_ms": np.array(iters["prefix_build_ms"]),
            "satisfaction_check_ms": np.array(iters["satisfaction_check_ms"]),
            "reduction_ms": np.array(iters["reduction_ms"]),
        }
        
        # 获取总迭代次数
        n_total = len(data["iteration"])
        n_show = min(20, n_total)
        
        print(f"文件: {result_file}")
        print(f"总迭代次数: {n_total}")
        print(f"显示前 {n_show} 次迭代\n")
        
        # 打印表头
        header = f"{'迭代':>6} | {'新增可达':>10} | {'新增认证':>10} | {'累计可达':>10} | {'累计认证':>10} | {'迭代耗时(ms)':>14} | {'前缀构建':>10} | {'满足检查':>10} | {'输入归约':>10}"
        separator = "-" * len(header)
        print(separator)
        print(header)
        print(separator)
        
        # 打印前20次迭代的数据
        for i in range(n_show):
            iter_num = data["iteration"][i]
            newly_rec = data["newly_reachable"][i]
            newly_cert = data["newly_certified"][i]
            total_rec = data["total_reachable"][i]
            total_cert = data["total_certified"][i]
            iter_time = data["iteration_ms"][i]
            prefix_time = data["prefix_build_ms"][i]
            sat_time = data["satisfaction_check_ms"][i]
            red_time = data["reduction_ms"][i]
            
            print(f"{int(iter_num):>6} | {int(newly_rec):>10} | {int(newly_cert):>10} | {int(total_rec):>10} | {int(total_cert):>10} | {iter_time:>14.3f} | {prefix_time:>10.3f} | {sat_time:>10.3f} | {red_time:>10.3f}")
        
        print(separator)
        
        # 打印汇总信息
        if n_show < n_total:
            print(f"\n... 共 {n_total} 次迭代，仅显示前 {n_show} 次")
        
        # 打印统计信息
        print(f"\n统计信息:")
        print(f"  平均迭代耗时: {np.mean(data['iteration_ms']):.2f} ms")
        print(f"  总认证状态数: {data['total_certified'][-1]}")
        print(f"  总可达状态数: {data['total_reachable'][-1]}")
        
        # 读取摘要信息
        if "summary/gpu" in f:
            summary = f["summary/gpu"]
            print(f"\nGPU求解摘要:")
            print(f"  收敛状态: {bool(summary['converged'][()])}")
            print(f"  消息: {summary['message'][()]}")
            print(f"  抽象阶段: {float(summary['abstraction_ms'][()]):.2f} ms")
            print(f"  求解阶段: {float(summary['solve_ms'][()]):.2f} ms")
    
    return 0

if __name__ == "__main__":
    exit(main())
