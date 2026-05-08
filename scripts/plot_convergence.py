#!/usr/bin/env python3
"""
收敛和性能可视化脚本

用法:
    python scripts/plot_convergence.py results/case_gpu.h5
    python scripts/plot_convergence.py results/case_gpu.h5 --output convergence.png
    python scripts/plot_convergence.py results/case_gpu.h5 --show-all
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_iteration_stats(h5_file):
    """从HDF5文件加载逐迭代统计数据"""
    if "iterations" not in h5_file:
        raise ValueError("HDF5文件中没有找到 'iterations' 组，可能不是GPU结果文件")
    
    iters = h5_file["iterations"]
    stats = {
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
    return stats


def load_summary(h5_file):
    """加载汇总信息"""
    summary = h5_file["summary/gpu"]
    return {
        "iterations": int(summary["iterations"][()]),
        "converged": bool(summary["converged"][()]),
        "message": summary["message"][()].decode("utf-8") if isinstance(summary["message"][()], bytes) else str(summary["message"][()]),
        "reachable_states": int(summary["reachable_states"][()]),
        "certified_candidate_states": int(summary["certified_candidate_states"][()]),
        "abstraction_ms": float(summary["abstraction_ms"][()]),
        "solve_ms": float(summary["solve_ms"][()]),
        "pair_count": int(summary["pair_count"][()]),
    }


def load_timings(h5_file):
    """加载kernel级别计时"""
    if "timings" not in h5_file:
        return None
    
    timings = h5_file["timings"]
    return {
        "abstraction_kernel_ms": float(timings["abstraction_kernel_ms"][()]),
        "scatter_mask_ms": float(timings["scatter_mask_ms"][()]),
        "scan_axis0_ms": float(timings["scan_axis0_ms"][()]),
        "scan_axis1_ms": float(timings["scan_axis1_ms"][()]),
        "scan_axis2_ms": float(timings["scan_axis2_ms"][()]),
        "scan_axis3_ms": float(timings["scan_axis3_ms"][()]),
        "pair_satisfaction_ms": float(timings["pair_satisfaction_ms"][()]),
        "reduce_inputs_ms": float(timings["reduce_inputs_ms"][()]),
        "memcpy_h2d_ms": float(timings["memcpy_h2d_ms"][()]),
        "memcpy_d2h_ms": float(timings["memcpy_d2h_ms"][()]),
    }


def plot_convergence(stats, summary, output_path=None):
    """绘制收敛曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"GPU求解收敛分析 (总迭代: {summary['iterations']}, 收敛: {summary['converged']})", 
                 fontsize=14, fontweight='bold')
    
    # 1. 累计认证候选数
    ax = axes[0, 0]
    ax.plot(stats["iteration"], stats["total_certified"], 'b-', linewidth=2, label="累计认证候选数")
    ax.axhline(y=summary["certified_candidate_states"], color='r', linestyle='--', 
               label=f"最终认证数: {summary['certified_candidate_states']}")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("认证候选状态数")
    ax.set_title("候选状态认证进度")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. 每次迭代新增认证数
    ax = axes[0, 1]
    ax.bar(stats["iteration"], stats["newly_certified"], color='green', alpha=0.7, label="新增认证数")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("新增认证状态数")
    ax.set_title("每次迭代新增认证候选数")
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # 3. 累计可达状态数
    ax = axes[1, 0]
    ax.plot(stats["iteration"], stats["total_reachable"], 'purple', linewidth=2, label="累计可达状态数")
    ax.axhline(y=summary["reachable_states"], color='orange', linestyle='--', 
               label=f"最终可达数: {summary['reachable_states']}")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("可达状态数")
    ax.set_title("可达集扩展进度")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. 每次迭代新增可达数
    ax = axes[1, 1]
    ax.bar(stats["iteration"], stats["newly_reachable"], color='cyan', alpha=0.7, label="新增可达数")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("新增可达状态数")
    ax.set_title("每次迭代新增可达状态数")
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"收敛图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_timing_breakdown(stats, summary, timings, output_path=None):
    """绘制性能分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"GPU性能分析 (抽象: {summary['abstraction_ms']:.1f}ms, 求解: {summary['solve_ms']:.1f}ms)", 
                 fontsize=14, fontweight='bold')
    
    # 1. 每次迭代总耗时
    ax = axes[0, 0]
    ax.plot(stats["iteration"], stats["iteration_ms"], 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("耗时 (ms)")
    ax.set_title("每次迭代总耗时")
    ax.grid(True, alpha=0.3)
    
    # 2. 迭代内部耗时分解（堆叠柱状图）
    ax = axes[0, 1]
    width = 0.8
    x = stats["iteration"]
    ax.bar(x, stats["prefix_build_ms"], width, label="前缀和构建", color='skyblue')
    ax.bar(x, stats["satisfaction_check_ms"], width, bottom=stats["prefix_build_ms"], 
           label="满足性检查", color='lightgreen')
    bottom = stats["prefix_build_ms"] + stats["satisfaction_check_ms"]
    ax.bar(x, stats["reduction_ms"], width, bottom=bottom, label="输入归约", color='salmon')
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("耗时 (ms)")
    ax.set_title("迭代内部耗时分解")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 平均迭代耗时分解（饼图）
    ax = axes[1, 0]
    avg_prefix = np.mean(stats["prefix_build_ms"])
    avg_satisfaction = np.mean(stats["satisfaction_check_ms"])
    avg_reduction = np.mean(stats["reduction_ms"])
    sizes = [avg_prefix, avg_satisfaction, avg_reduction]
    labels = [f"前缀和构建\n{avg_prefix:.2f}ms", 
              f"满足性检查\n{avg_satisfaction:.2f}ms", 
              f"输入归约\n{avg_reduction:.2f}ms"]
    colors = ['skyblue', 'lightgreen', 'salmon']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title("平均迭代耗时分解")
    
    # 4. Kernel级别计时（如果有）
    ax = axes[1, 1]
    if timings:
        kernel_names = ["抽象kernel", "scatter", "scan_x", "scan_y", "scan_z", "scan_w", 
                       "pair_sat", "reduce", "H2D", "D2H"]
        kernel_times = [
            timings["abstraction_kernel_ms"],
            timings["scatter_mask_ms"],
            timings["scan_axis0_ms"],
            timings["scan_axis1_ms"],
            timings["scan_axis2_ms"],
            timings["scan_axis3_ms"],
            timings["pair_satisfaction_ms"],
            timings["reduce_inputs_ms"],
            timings["memcpy_h2d_ms"],
            timings["memcpy_d2h_ms"],
        ]
        # 只显示非零的kernel
        non_zero = [(n, t) for n, t in zip(kernel_names, kernel_times) if t > 0]
        if non_zero:
            names, times = zip(*non_zero)
            y_pos = np.arange(len(names))
            ax.barh(y_pos, times, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlabel("耗时 (ms)")
            ax.set_title("Kernel级别计时")
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, "无Kernel计时数据", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, "无Kernel计时数据", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"性能分析图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_efficiency_metrics(stats, summary, output_path=None):
    """绘制效率指标"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GPU求解效率指标", fontsize=14, fontweight='bold')
    
    # 1. 每毫秒认证的候选数（吞吐量）
    ax = axes[0]
    throughput = stats["newly_certified"] / stats["iteration_ms"]
    throughput[np.isnan(throughput)] = 0  # 处理除零
    ax.plot(stats["iteration"], throughput, 'g-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("认证吞吐量 (状态/ms)")
    ax.set_title("每毫秒认证候选数")
    ax.grid(True, alpha=0.3)
    
    # 2. 累计效率（总认证数 / 累计时间）
    ax = axes[1]
    cumulative_time = np.cumsum(stats["iteration_ms"])
    cumulative_efficiency = stats["total_certified"] / cumulative_time
    ax.plot(stats["iteration"], cumulative_efficiency, 'b-', linewidth=2)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("累计效率 (状态/ms)")
    ax.set_title("累计认证效率")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"效率指标图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_stats(stats, summary, timings):
    """打印统计摘要"""
    print("\n" + "="*60)
    print("GPU求解统计摘要")
    print("="*60)
    print(f"总迭代次数: {summary['iterations']}")
    print(f"收敛状态: {'已收敛' if summary['converged'] else '未收敛'}")
    print(f"消息: {summary['message']}")
    print(f"最终认证候选数: {summary['certified_candidate_states']}")
    print(f"最终可达状态数: {summary['reachable_states']}")
    print(f"状态-输入对总数: {summary['pair_count']}")
    print(f"\n抽象阶段耗时: {summary['abstraction_ms']:.2f} ms")
    print(f"求解阶段耗时: {summary['solve_ms']:.2f} ms")
    print(f"总耗时: {summary['abstraction_ms'] + summary['solve_ms']:.2f} ms")
    
    if len(stats["iteration"]) > 0:
        print(f"\n平均每次迭代耗时: {np.mean(stats['iteration_ms']):.2f} ms")
        print(f"  - 前缀和构建: {np.mean(stats['prefix_build_ms']):.2f} ms ({np.mean(stats['prefix_build_ms'])/np.mean(stats['iteration_ms'])*100:.1f}%)")
        print(f"  - 满足性检查: {np.mean(stats['satisfaction_check_ms']):.2f} ms ({np.mean(stats['satisfaction_check_ms'])/np.mean(stats['iteration_ms'])*100:.1f}%)")
        print(f"  - 输入归约: {np.mean(stats['reduction_ms']):.2f} ms ({np.mean(stats['reduction_ms'])/np.mean(stats['iteration_ms'])*100:.1f}%)")
        
        total_newly_certified = np.sum(stats["newly_certified"])
        total_time = np.sum(stats["iteration_ms"])
        print(f"\n总认证吞吐量: {total_newly_certified / total_time:.2f} 状态/ms")
        print(f"平均每次迭代认证: {np.mean(stats['newly_certified']):.2f} 状态")
    
    if timings and timings["abstraction_kernel_ms"] > 0:
        print(f"\nKernel级别计时:")
        print(f"  - 抽象kernel: {timings['abstraction_kernel_ms']:.2f} ms")
        if timings["memcpy_h2d_ms"] > 0:
            print(f"  - Host→Device传输: {timings['memcpy_h2d_ms']:.2f} ms")
        if timings["memcpy_d2h_ms"] > 0:
            print(f"  - Device→Host传输: {timings['memcpy_d2h_ms']:.2f} ms")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="可视化GPU求解的收敛过程和性能数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s results/case_gpu.h5
  %(prog)s results/case_gpu.h5 --output convergence.png
  %(prog)s results/case_gpu.h5 --show-all --output-dir plots/
        """
    )
    parser.add_argument("input", type=str, help="输入HDF5文件路径")
    parser.add_argument("--output", type=str, help="输出图像文件路径（默认显示）")
    parser.add_argument("--output-dir", type=str, help="输出目录（生成多个图像文件）")
    parser.add_argument("--show-all", action="store_true", help="生成所有可视化图表")
    parser.add_argument("--no-summary", action="store_true", help="不打印统计摘要")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}", file=sys.stderr)
        return 1
    
    try:
        with h5py.File(input_path, "r") as f:
            stats = load_iteration_stats(f)
            summary = load_summary(f)
            timings = load_timings(f)
        
        if not args.no_summary:
            print_summary_stats(stats, summary, timings)
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plot_convergence(stats, summary, output_dir / "convergence.png")
            plot_timing_breakdown(stats, summary, timings, output_dir / "timing.png")
            plot_efficiency_metrics(stats, summary, output_dir / "efficiency.png")
            
            print(f"\n所有图表已保存到: {output_dir}/")
        elif args.show_all:
            plot_convergence(stats, summary)
            plot_timing_breakdown(stats, summary, timings)
            plot_efficiency_metrics(stats, summary)
        else:
            # 默认只显示收敛图
            plot_convergence(stats, summary, args.output)
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
