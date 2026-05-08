function h = plotReachSet(kinv, options)
% plotReachSet - 绘制鲁棒可达集的3D等值面
%
% 该函数使用 isosurface 绘制 K-Invariant 可达集的3D可视化。
% 可达集内的状态（reachSetRobust < inf）将被渲染为半透明等值面。
%
% 用法:
%   h = symctrl.visualization.plotReachSet(kinv)
%   h = symctrl.visualization.plotReachSet(kinv, 'FaceColor', 'blue')
%   h = symctrl.visualization.plotReachSet(kinv, 'FaceAlpha', 0.3, 'IsoValue', 0.5)
%
% 输入:
%   kinv    - KInvariant 对象
%   options - 可选参数:
%       FaceColor  - 表面颜色 (默认 'red')
%       FaceAlpha  - 透明度 0~1 (默认 0.2)
%       EdgeColor  - 边缘颜色 (默认 'none')
%       IsoValue   - 等值面阈值 (默认 0.5)
%       ViewAngle  - 视角 [azimuth, elevation] (默认 [100, 16])
%       ShowAxis   - 是否设置坐标轴 (默认 true)
%       Lighting   - 是否启用光照 (默认 true)
%
% 输出:
%   h - patch 图形句柄
%
% 示例:
%   kinv = symctrl.KInvariant.load('results/kinv_unicycle_robust.mat');
%   figure;
%   h = symctrl.visualization.plotReachSet(kinv, 'FaceColor', 'blue', 'FaceAlpha', 0.3);

arguments
    kinv
    options.FaceColor = 'red'
    options.FaceAlpha = 0.2
    options.EdgeColor = 'none'
    options.IsoValue = 0.5
    options.ViewAngle = [100, 16]
    options.ShowAxis = true
    options.Lighting = true
end

% 提取状态空间信息
ss = kinv.stateSpace;
bound = ss.bound;
n_x = ss.n_x;

% 将可达集从1D向量重塑为3D数组
% reachSetRobust: inf表示不可达，有限值表示可达
reachSetRobust = kinv.reachSetRobust;

% 创建指示函数: 1表示可达，0表示不可达
I = double(reachSetRobust ~= inf);

% 重塑为3D数组 (注意MATLAB的索引顺序)
I_3d = reshape(I, n_x');

% 创建网格坐标
[xx, yy, zz] = meshgrid(...
    linspace(bound(1,1), bound(1,2), n_x(1)), ...
    linspace(bound(2,1), bound(2,2), n_x(2)), ...
    linspace(bound(3,1), bound(3,2), n_x(3)));

% 绘制等值面 (需要permute以匹配meshgrid的输出顺序)
I_permuted = permute(I_3d, [2 1 3]);
[faces, verts] = isosurface(xx, yy, zz, I_permuted, options.IsoValue);

% 检查是否有有效的等值面
if isempty(verts) || isempty(faces)
    % 创建一个不可见的占位符 patch
    h = patch('Faces', [1 2 3], 'Vertices', [0 0 0; 0 0 0; 0 0 0], ...
              'FaceColor', options.FaceColor, 'FaceAlpha', 0, ...
              'EdgeColor', 'none', 'Visible', 'off');
    warning('plotReachSet: No valid isosurface generated (reachable set may be empty or too small)');
    return;
end

h = patch('Faces', faces, 'Vertices', verts);

% 计算法向量以改善渲染效果
isonormals(xx, yy, zz, I_permuted, h);

% 设置外观属性
h.FaceAlpha = options.FaceAlpha;
h.FaceColor = options.FaceColor;
h.EdgeColor = options.EdgeColor;

% 设置坐标轴
if options.ShowAxis
    daspect([1, 1, 1]);
    view(options.ViewAngle);
    
    % 设置坐标轴范围 (稍微扩展边界)
    margin = [0.5; 0.5; 0.2];
    axis([bound(1,1)-margin(1), bound(1,2)+margin(1), ...
          bound(2,1)-margin(2), bound(2,2)+margin(2), ...
          bound(3,1)-margin(3), bound(3,2)+margin(3)]);
    
    xlabel('x_1');
    ylabel('x_2');
    zlabel('\theta', 'Interpreter', 'latex');
    grid on;
end

% 添加光照效果
if options.Lighting
    camlight headlight;
    lighting gouraud;
end

end
