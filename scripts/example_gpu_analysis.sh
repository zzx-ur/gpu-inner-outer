#!/bin/bash
# GPU求解和可视化分析示例脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== GPU符号控制求解和分析示例 ===${NC}\n"

# 检查构建
if [ ! -f "build/cuda_symbolic_control_cli" ]; then
    echo "错误: 未找到可执行文件 build/cuda_symbolic_control_cli"
    echo "请先运行: cmake -B build -S . && cmake --build build"
    exit 1
fi

# 检查示例配置
if [ ! -f "examples/hyperbolic_case.cfg" ]; then
    echo "错误: 未找到示例配置文件 examples/hyperbolic_case.cfg"
    exit 1
fi

# 创建输出目录
mkdir -p results
mkdir -p plots

# 运行GPU求解
echo -e "${GREEN}步骤 1: 运行GPU求解${NC}"
./build/cuda_symbolic_control_cli examples/hyperbolic_case.cfg --gpu --output results/hyperbolic_gpu.h5

echo -e "\n${GREEN}步骤 2: 生成收敛和性能可视化${NC}"
python3 scripts/plot_convergence.py results/hyperbolic_gpu.h5 --show-all --output-dir plots/

echo -e "\n${GREEN}步骤 3: 生成3D可达集可视化${NC}"
python3 scripts/plot_results.py results/hyperbolic_gpu.h5 --plot-mode sets3d --output plots/reachable_set_3d.png

echo -e "\n${BLUE}=== 分析完成 ===${NC}"
echo "结果文件: results/hyperbolic_gpu.h5"
echo "可视化图表:"
echo "  - plots/convergence.png    (收敛曲线)"
echo "  - plots/timing.png         (性能分析)"
echo "  - plots/efficiency.png     (效率指标)"
echo "  - plots/reachable_set_3d.png (3D可达集)"
echo ""
echo "查看详细统计:"
echo "  python3 scripts/plot_convergence.py results/hyperbolic_gpu.h5"
