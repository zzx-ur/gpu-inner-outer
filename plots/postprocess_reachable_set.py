#!/usr/bin/env python3
"""
后处理脚本：从符号控制结果中提取可达集合并保存为MATLAB格式
功能：
  1. 从 hyperbolic_demo_gpu.h5 加载可达集合掩码和网格参数
  2. 将一维掩码重塑为4D网格
  3. 投影速度维度得到3D可达集合
  4. 提取等值面顶点和面片
  5. 映射到物理坐标系
  6. 导出MATLAB格式结果
"""

import h5py
import numpy as np
from scipy.io import savemat
from scipy.ndimage import binary_dilation
import warnings

warnings.filterwarnings('ignore')


class ReachableSetExtractor:
    """可达集合提取器"""
    
    def __init__(self, h5_path):
        """
        初始化提取器，加载h5文件中的可达集合和网格参数
        
        Args:
            h5_path: hyperbolic_demo_gpu.h5 文件路径
        """
        self.h5_path = h5_path
        self._load_data()
    
    def _load_data(self):
        """从h5文件加载可达集合和网格参数"""
        with h5py.File(self.h5_path, 'r') as f:
            # 加载可达集合掩码（一维数组）
            self.reachable_mask = f['result/gpu/reachable_mask'][:]
            self.candidate_mask = f['result/gpu/candidate_mask'][:]
            self.reach_step = f['result/gpu/reach_step'][:]
            
            # 加载状态空间网格参数
            self.state_lb = f['state/lb'][:]
            self.state_ub = f['state/ub'][:]
            self.state_eta = f['state/eta'][:]
            self.state_shape = f['state/shape'][:]
            self.state_strides = f['state/strides'][:]
            
            # 加载配置参数
            self.candidate_lb = f['config/candidate_lb'][:]
            self.candidate_ub = f['config/candidate_ub'][:]
            
            # 加载元数据
            self.case_name = f['meta/case_name'][()].decode('utf-8')
        
        print(f"✓ 已加载可达集合掩码，大小: {self.reachable_mask.shape}")
        print(f"✓ 状态空间形状: {self.state_shape}")
        print(f"✓ 候选集合范围: {self.candidate_lb} -> {self.candidate_ub}")
    
    def extract_reachable_set_3d(self):
        """
        将可达集合从1D掩码转换为3D物理坐标
        
        Returns:
            reachable_points: (N, 3) 数组，包含所有可达点的物理坐标 [x, y, theta]
            reachable_mask_3d: (shape[1], shape[2], shape[3]) 3D掩码（投影速度维度）
        """
        # 重塑为4D网格
        reachable_4d = self.reachable_mask.reshape(
            self.state_shape[0], self.state_shape[1], 
            self.state_shape[2], self.state_shape[3]
        )
        
        # 交换维度顺序以匹配MATLAB约定 [y, x, theta, speed]
        reachable_4d = np.transpose(reachable_4d, [1, 0, 2, 3])
        
        # 投影速度维度：如果任何速度值为可达，则该(x,y,theta)点可达
        reachable_3d = np.any(reachable_4d != 0, axis=3)
        
        # 转置回 [x, y, theta] 顺序用于后续处理
        reachable_3d = np.transpose(reachable_3d, [1, 0, 2])
        
        print(f"✓ 可达集合3D投影完成，非零点数: {np.sum(reachable_3d)}")
        
        return reachable_3d
    
    def extract_isosurface_vertices(self, reachable_3d):
        """
        从3D可达集合提取等值面顶点
        
        Args:
            reachable_3d: (shape[0], shape[1], shape[2]) 3D掩码
        
        Returns:
            vertices_phys: (N, 3) 物理坐标顶点
            faces: (M, 3) 三角形面片索引
        """
        try:
            from skimage import measure
            
            # 使用marching cubes算法提取等值面
            verts_idx, faces, _, _ = measure.marching_cubes(
                reachable_3d.astype(float),
                level=0.5,
                spacing=(self.state_eta[0], self.state_eta[1], self.state_eta[2])
            )
            
            # 映射到物理坐标
            vertices_phys = np.zeros_like(verts_idx)
            vertices_phys[:, 0] = self.state_lb[0] + verts_idx[:, 0]  # x
            vertices_phys[:, 1] = self.state_lb[1] + verts_idx[:, 1]  # y
            vertices_phys[:, 2] = self.state_lb[2] + verts_idx[:, 2]  # theta
            
            print(f"✓ 等值面提取完成，顶点数: {len(vertices_phys)}, 面片数: {len(faces)}")
            
            return vertices_phys, faces
        
        except ImportError:
            print("⚠ scikit-image未安装，使用简化方法提取顶点")
            return self._extract_vertices_simple(reachable_3d)
    
    def _extract_vertices_simple(self, reachable_3d):
        """
        简化方法：直接从掩码提取所有可达点作为顶点
        
        Args:
            reachable_3d: 3D掩码
        
        Returns:
            vertices_phys: 物理坐标顶点
            faces: 空面片数组（需要在MATLAB中处理）
        """
        # 获取所有可达点的索引
        idx = np.where(reachable_3d)
        
        # 转换为物理坐标
        vertices_phys = np.zeros((len(idx[0]), 3))
        vertices_phys[:, 0] = self.state_lb[0] + idx[0] * self.state_eta[0]  # x
        vertices_phys[:, 1] = self.state_lb[1] + idx[1] * self.state_eta[1]  # y
        vertices_phys[:, 2] = self.state_lb[2] + idx[2] * self.state_eta[2]  # theta
        
        print(f"✓ 简化方法提取完成，顶点数: {len(vertices_phys)}")
        
        return vertices_phys, np.array([])
    
    def extract_candidate_set(self):
        """
        提取候选集合的8个顶点（矩形棱柱）
        
        Returns:
            candidate_verts: (8, 3) 候选集合顶点
        """
        candidate_verts = np.array([
            [self.candidate_lb[0], self.candidate_lb[1], self.candidate_lb[2]],
            [self.candidate_ub[0], self.candidate_lb[1], self.candidate_lb[2]],
            [self.candidate_ub[0], self.candidate_ub[1], self.candidate_lb[2]],
            [self.candidate_lb[0], self.candidate_ub[1], self.candidate_lb[2]],
            [self.candidate_lb[0], self.candidate_lb[1], self.candidate_ub[2]],
            [self.candidate_ub[0], self.candidate_lb[1], self.candidate_ub[2]],
            [self.candidate_ub[0], self.candidate_ub[1], self.candidate_ub[2]],
            [self.candidate_lb[0], self.candidate_ub[1], self.candidate_ub[2]],
        ])
        
        print(f"✓ 候选集合顶点提取完成")
        
        return candidate_verts
    
    def export_to_matlab(self, output_path, reachable_3d, vertices_phys, faces, candidate_verts):
        """
        将提取的可达集合导出为MATLAB格式
        
        Args:
            output_path: 输出文件路径
            reachable_3d: 3D可达集合掩码
            vertices_phys: 等值面顶点
            faces: 面片索引
            candidate_verts: 候选集合顶点
        """
        matlab_data = {
            'reachable_mask_3d': reachable_3d.astype(np.uint8),
            'reachable_vertices': vertices_phys,
            'reachable_faces': faces.astype(np.int32) if len(faces) > 0 else np.array([]),
            'candidate_vertices': candidate_verts,
            'candidate_faces': np.array([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [1, 2, 6, 5],
                [3, 4, 8, 7],
                [1, 4, 8, 5],
                [2, 3, 7, 6]
            ], dtype=np.int32),
            'state_lb': self.state_lb,
            'state_ub': self.state_ub,
            'state_eta': self.state_eta,
            'state_shape': self.state_shape.astype(np.int32),
            'candidate_lb': self.candidate_lb,
            'candidate_ub': self.candidate_ub,
        }
        
        savemat(output_path, matlab_data)
        print(f"\n✓ 结果已导出到: {output_path}")


def main():
    """主函数"""
    import os
    
    # 文件路径（使用相对路径以支持远程执行）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    h5_path = os.path.join(project_root, 'results', 'hyperbolic_demo_gpu.h5')
    output_dir = script_dir
    output_path = os.path.join(output_dir, 'reachable_set.mat')
    
    # 检查h5文件是否存在
    if not os.path.exists(h5_path):
        print(f"错误: 找不到文件 {h5_path}")
        return
    
    # 创建提取器
    extractor = ReachableSetExtractor(h5_path)
    
    # 提取可达集合
    print("\n开始提取可达集合...")
    reachable_3d = extractor.extract_reachable_set_3d()
    
    # 提取等值面
    print("\n提取等值面...")
    vertices_phys, faces = extractor.extract_isosurface_vertices(reachable_3d)
    
    # 提取候选集合
    print("\n提取候选集合...")
    candidate_verts = extractor.extract_candidate_set()
    
    # 导出MATLAB格式
    print("\n导出MATLAB格式...")
    extractor.export_to_matlab(output_path, reachable_3d, vertices_phys, faces, candidate_verts)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("提取统计信息")
    print("=" * 60)
    print(f"可达点总数: {np.sum(reachable_3d)}")
    print(f"等值面顶点数: {len(vertices_phys)}")
    print(f"等值面面片数: {len(faces)}")
    print(f"状态空间范围: {extractor.state_lb} -> {extractor.state_ub}")
    print(f"候选集合范围: {extractor.candidate_lb} -> {extractor.candidate_ub}")


if __name__ == '__main__':
    main()
