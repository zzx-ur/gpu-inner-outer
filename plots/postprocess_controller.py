#!/usr/bin/env python3
"""
后处理脚本：从符号控制结果中提取控制器并进行闭环轨迹仿真
功能：
  1. 从 hyperbolic_demo_gpu.h5 加载控制器和网格参数
  2. 实现连续状态到离散网格索引的映射
  3. 通过查表获取控制输入
  4. 进行4D动力学积分（欧拉法）
  5. 生成两条独立的轨迹并注入不同的随机扰动
  6. 导出MATLAB格式结果
"""

import h5py
import numpy as np
from scipy.io import savemat
import warnings


class ControllerSimulator:
    """控制器仿真器"""
    
    def __init__(self, h5_path):
        """
        初始化仿真器，加载h5文件中的控制器和网格参数
        
        Args:
            h5_path: hyperbolic_demo_gpu.h5 文件路径
        """
        self.h5_path = h5_path
        self._load_data()
    
    def _load_data(self):
        """从h5文件加载控制器和网格参数"""
        with h5py.File(self.h5_path, 'r') as f:
            # 加载控制器（一维数组）
            self.controller = f['result/gpu/controller'][:]
            
            # 加载状态空间网格参数
            self.state_lb = f['state/lb'][:]
            self.state_ub = f['state/ub'][:]
            self.state_eta = f['state/eta'][:]
            self.state_shape = f['state/shape'][:]
            self.state_strides = f['state/strides'][:]
            self.state_wrap_dim = int(f['state/wrap_dim'][()])
            
            # 加载输入空间网格参数
            self.input_lb = f['input/lb'][:]
            self.input_ub = f['input/ub'][:]
            self.input_eta = f['input/eta'][:]
            self.input_shape = f['input/shape'][:]
            self.input_centers = f['input/centers'][:]
            
            # 加载配置参数
            self.sample_time = float(f['config/sample_time'][()])
            self.integration_substeps = int(f['config/integration_substeps'][()])
            self.disturbance_half_width = f['config/disturbance_half_width'][:]
            
            # 加载元数据
            self.invalid_input = int(f['meta/invalid_input'][()])
        
        print(f"✓ 已加载控制器，大小: {self.controller.shape}")
        print(f"✓ 状态空间形状: {self.state_shape}")
        print(f"✓ 输入空间形状: {self.input_shape}")
        print(f"✓ 采样时间: {self.sample_time}s, 积分子步: {self.integration_substeps}")
    
    def state_to_flat_index(self, state):
        """
        将4D连续状态映射到一维网格索引
        
        Args:
            state: 4D连续状态向量 [x, y, theta, v]
        
        Returns:
            flat_index: 一维网格索引，若超出范围返回None
        """
        state = np.array(state, dtype=np.float64)
        
        # 对第wrap_dim维（角度）进行周期性限制到[-π, π]
        state[self.state_wrap_dim] = np.arctan2(
            np.sin(state[self.state_wrap_dim]),
            np.cos(state[self.state_wrap_dim])
        )
        
        # 计算多维网格索引
        multi_idx = np.zeros(4, dtype=np.int64)
        for i in range(4):
            # 从连续值映射到网格索引
            normalized = (state[i] - self.state_lb[i]) / self.state_eta[i]
            multi_idx[i] = int(np.round(normalized))
            
            # 边界检查（允许轻微超出，通过clamp处理）
            if multi_idx[i] < 0:
                multi_idx[i] = 0
            elif multi_idx[i] >= self.state_shape[i]:
                multi_idx[i] = self.state_shape[i] - 1
        
        # 展平为一维索引
        flat_index = int(np.dot(multi_idx, self.state_strides))
        
        if flat_index < 0 or flat_index >= len(self.controller):
            return None
        
        return flat_index
    
    def get_control_input(self, state):
        """
        通过查表获取控制输入
        
        Args:
            state: 4D连续状态向量
        
        Returns:
            u: 2D控制输入向量 [a, omega]，若无效返回None
        """
        flat_idx = self.state_to_flat_index(state)
        
        if flat_idx is None:
            return None
        
        # 获取离散输入索引
        input_idx = self.controller[flat_idx]
        
        # 检查是否为无效标记
        if input_idx == self.invalid_input:
            return None
        
        # 逆向映射回连续控制输入
        if input_idx >= len(self.input_centers):
            return None
        
        u = self.input_centers[input_idx].copy()
        return u
    
    def update_state(self, x, u, disturbance, dt_substep=None):
        """
        4D动力学状态更新函数（单个子步）
        
        动力学模型：
          dx/dt = v * cos(theta)
          dy/dt = v * sin(theta)
          dtheta/dt = omega
          dv/dt = a
        
        Args:
            x: 当前状态 [x, y, theta, v]
            u: 控制输入 [a, omega]
            disturbance: 扰动向量 [w_x, w_y, w_theta, w_v]
            dt_substep: 单个子步的时间步长
        
        Returns:
            x_new: 更新后的状态
        """
        if dt_substep is None:
            dt_substep = self.sample_time / self.integration_substeps
        
        x = np.array(x, dtype=np.float64)
        u = np.array(u, dtype=np.float64)
        disturbance = np.array(disturbance, dtype=np.float64)
        
        # 4D动力学
        dx_dt = np.zeros(4)
        dx_dt[0] = x[3] * np.cos(x[2])  # dx/dt = v * cos(theta)
        dx_dt[1] = x[3] * np.sin(x[2])  # dy/dt = v * sin(theta)
        dx_dt[2] = u[1]                  # dtheta/dt = omega
        dx_dt[3] = u[0]                  # dv/dt = a
        
        # 欧拉积分 + 扰动注入
        x_new = x + dt_substep * (dx_dt + disturbance)
        
        # 对角度进行周期性限制
        x_new[2] = np.arctan2(np.sin(x_new[2]), np.cos(x_new[2]))
        
        return x_new
    
    def simulate_trajectory(self, x0, N, seed=None):
        """
        仿真单条轨迹
        
        Args:
            x0: 初始状态 [x, y, theta, v]
            N: 仿真步数
            seed: 随机数种子（用于可重复性）
        
        Returns:
            trajectory: 形状为 (N+1, 4) 的轨迹数组
            disturbances: 形状为 (N, integration_substeps, 4) 的扰动序列
        """
        if seed is not None:
            np.random.seed(seed)
        
        trajectory = np.zeros((N + 1, 4))
        trajectory[0] = x0
        
        # 记录所有扰动
        disturbances = np.zeros((N, self.integration_substeps, 4))
        
        dt_substep = self.sample_time / self.integration_substeps
        
        for step in range(N):
            x_current = trajectory[step].copy()
            
            # 获取控制输入
            u = self.get_control_input(x_current)
            if u is None:
                # 无法获取控制输入，停止仿真
                break
            
            # 在一个采样周期内进行多个子步积分
            for substep in range(self.integration_substeps):
                # 生成随机扰动（均匀分布）
                disturbance = np.random.uniform(
                    -self.disturbance_half_width,
                    self.disturbance_half_width,
                    size=4
                )
                disturbances[step, substep] = disturbance
                
                # 更新状态
                x_current = self.update_state(x_current, u, disturbance, dt_substep)
            
            trajectory[step + 1] = x_current
        
        return trajectory, disturbances
    
    def run_dual_trajectory_simulation(self, x0, N):
        """
        运行双轨迹对比仿真
        
        Args:
            x0: 初始状态
            N: 仿真步数
        
        Returns:
            results: 包含两条轨迹、时间数组和扰动序列的字典
        """
        print(f"\n开始双轨迹仿真...")
        print(f"初始状态: {x0}")
        print(f"仿真步数: {N}")
        
        # 轨迹1（种子1）
        print("\n[轨迹1] 仿真中...")
        traj_1, dist_1 = self.simulate_trajectory(x0, N, seed=42)
        print(f"✓ 轨迹1完成，最终状态: {traj_1[-1]}")
        
        # 轨迹2（种子2）
        print("\n[轨迹2] 仿真中...")
        traj_2, dist_2 = self.simulate_trajectory(x0, N, seed=123)
        print(f"✓ 轨迹2完成，最终状态: {traj_2[-1]}")
        
        # 时间数组
        time_array = np.arange(N + 1) * self.sample_time
        
        results = {
            'trajectory_1': traj_1,
            'trajectory_2': traj_2,
            'time': time_array,
            'disturbances_1': dist_1,
            'disturbances_2': dist_2,
            'x0': x0,
            'N': N,
            'sample_time': self.sample_time,
            'integration_substeps': self.integration_substeps,
        }
        
        return results
    
    def export_to_matlab(self, results, output_path):
        """
        将仿真结果导出为MATLAB格式
        
        Args:
            results: 仿真结果字典
            output_path: 输出文件路径
        """
        # 准备MATLAB格式的数据
        matlab_data = {
            'trajectory_1': results['trajectory_1'],
            'trajectory_2': results['trajectory_2'],
            'time': results['time'],
            'x0': results['x0'],
            'N': np.array([results['N']]),
            'sample_time': np.array([results['sample_time']]),
            'integration_substeps': np.array([results['integration_substeps']]),
        }
        
        # 保存扰动序列（需要特殊处理因为维度不同）
        # 将扰动展平为2D数组便于MATLAB读取
        dist_1_flat = results['disturbances_1'].reshape(
            results['disturbances_1'].shape[0] * results['disturbances_1'].shape[1],
            results['disturbances_1'].shape[2]
        )
        dist_2_flat = results['disturbances_2'].reshape(
            results['disturbances_2'].shape[0] * results['disturbances_2'].shape[1],
            results['disturbances_2'].shape[2]
        )
        
        matlab_data['disturbances_1'] = dist_1_flat
        matlab_data['disturbances_2'] = dist_2_flat
        matlab_data['disturbance_shape'] = np.array([
            results['disturbances_1'].shape[0],
            results['disturbances_1'].shape[1],
            results['disturbances_1'].shape[2]
        ])
        
        savemat(output_path, matlab_data)
        print(f"\n✓ 结果已导出到: {output_path}")


def main():
    """主函数"""
    import os
    
    # 文件路径
    h5_path = '/Users/zhixin/Documents/symbol_control/kinv_gpu/results/hyperbolic_demo_gpu.h5'
    output_dir = '/Users/zhixin/Documents/symbol_control/kinv_gpu/plots'
    output_path = os.path.join(output_dir, 'simulation_results.mat')
    
    # 检查h5文件是否存在
    if not os.path.exists(h5_path):
        print(f"错误: 找不到文件 {h5_path}")
        return
    
    # 创建仿真器
    simulator = ControllerSimulator(h5_path)
    
    # 设置初始状态和仿真参数
    x0 = np.array([-0.8, -0.8, -2.0, 0.5])
    N = 40
    
    # 运行双轨迹仿真
    results = simulator.run_dual_trajectory_simulation(x0, N)
    
    # 导出MATLAB格式
    simulator.export_to_matlab(results, output_path)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("仿真统计信息")
    print("=" * 60)
    print(f"轨迹1 - 初始状态: {results['trajectory_1'][0]}")
    print(f"轨迹1 - 最终状态: {results['trajectory_1'][-1]}")
    print(f"轨迹1 - 位置变化: Δx={results['trajectory_1'][-1, 0] - results['trajectory_1'][0, 0]:.4f}, "
          f"Δy={results['trajectory_1'][-1, 1] - results['trajectory_1'][0, 1]:.4f}")
    
    print(f"\n轨迹2 - 初始状态: {results['trajectory_2'][0]}")
    print(f"轨迹2 - 最终状态: {results['trajectory_2'][-1]}")
    print(f"轨迹2 - 位置变化: Δx={results['trajectory_2'][-1, 0] - results['trajectory_2'][0, 0]:.4f}, "
          f"Δy={results['trajectory_2'][-1, 1] - results['trajectory_2'][0, 1]:.4f}")
    
    # 计算轨迹差异
    traj_diff = np.linalg.norm(results['trajectory_1'] - results['trajectory_2'], axis=1)
    print(f"\n轨迹差异统计:")
    print(f"  最大差异: {np.max(traj_diff):.6f}")
    print(f"  平均差异: {np.mean(traj_diff):.6f}")
    print(f"  最终差异: {traj_diff[-1]:.6f}")


if __name__ == '__main__':
    main()
