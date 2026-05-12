%% Visualize hyperbolic_demo_gpu.h5 - Candidate vs Reachable 3D Plot
%
% Usage: Open and run this script in MATLAB (or press F5)

clear; close all; clc;

%% Configuration
% Determine project root (parent of scripts/)
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

% 请确保 h5 文件路径正确
result_file = fullfile(project_root, 'results', 'hyperbolic_demo_gpu.h5');
run_label   = 'gpu';

% 优化了颜色搭配，使其对比更明显 (参考之前成功的配色)
candidate_color  = [0.2, 0.2, 0.6]; % 深紫蓝色
candidate_alpha  = 0.8;            % lowered transparency (was 0.6)
reachable_color  = [0.2, 0.2, 0.6]; % 浅红色
reachable_alpha  = 0.2;
azim             = 100;
elev             = 16;

%% Load data
fprintf('Loading %s (run: %s)...\n', result_file, run_label);

% Read metadata
shape = double(h5read(result_file, '/state/shape')');
lb    = h5read(result_file, '/state/lb')';
eta   = h5read(result_file, '/state/eta')';

% Read candidate bounds from config
candidate_lb = h5read(result_file, '/config/candidate_lb')';
candidate_ub = h5read(result_file, '/config/candidate_ub')';

cand_x_min = candidate_lb(1); cand_x_max = candidate_ub(1);
cand_y_min = candidate_lb(2); cand_y_max = candidate_ub(2);
cand_theta_min = candidate_lb(3); cand_theta_max = candidate_ub(3);

% Load candidate and reachable masks
run_path = sprintf('/result/%s', run_label);
candidate_raw = h5read(result_file, fullfile(run_path, 'candidate_mask'));
reachable_raw = h5read(result_file, fullfile(run_path, 'reachable_mask'));

%% Reshape to 4D grid 
candidate_4d = reshape(candidate_raw, shape(1), shape(2), shape(3), shape(4));
candidate_4d = permute(candidate_4d, [2 1 3 4]);  % [y, x, theta, speed]

reachable_4d = reshape(reachable_raw, shape(1), shape(2), shape(3), shape(4));
reachable_4d = permute(reachable_4d, [2 1 3 4]);

% Project speed dimension
candidate_xyz = any(candidate_4d ~= 0, 4);  
reachable_xyz = any(reachable_4d ~= 0, 4);

%% Extract isosurface for reachable set
reachable_vol = permute(reachable_xyz, [2 1 3]);
[reach_faces, reach_verts] = isosurface(reachable_vol, 0.5);

% Map reachable vertices to physical coordinates
reach_y = lb(2) + (reach_verts(:,1) - 0.5) * eta(2);
reach_x = lb(1) + (reach_verts(:,2) - 0.5) * eta(1);
reach_theta = lb(3) + (reach_verts(:,3) - 0.5) * eta(3);
reachable_verts_phys = [reach_x, reach_y, reach_theta];

%% Define candidate set as rectangular prism
C = [cand_x_min, cand_y_min, cand_theta_min;
     cand_x_max, cand_y_min, cand_theta_min;
     cand_x_max, cand_y_max, cand_theta_min;
     cand_x_min, cand_y_max, cand_theta_min;
     cand_x_min, cand_y_min, cand_theta_max;
     cand_x_max, cand_y_min, cand_theta_max;
     cand_x_max, cand_y_max, cand_theta_max;
     cand_x_min, cand_y_max, cand_theta_max];

cand_faces_rect = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 1 4 8 5; 2 3 7 6];
candidate_verts_phys = C;

%% Create plot
fig = figure('Color', 'white', 'Position', [100 100 1100 850]);
ax = axes('Parent', fig);
hold(ax, 'on');

% Plot candidate set
if exist('cand_faces_rect', 'var') && size(candidate_verts_phys, 1) >= 8
    patch(ax, 'Faces', cand_faces_rect, 'Vertices', candidate_verts_phys, ...
          'FaceColor', candidate_color, 'FaceAlpha', candidate_alpha, ...
          'EdgeColor', 'k', 'LineWidth', 1.5);
end

% Plot reachable set
if ~isempty(reach_faces)
    patch(ax, 'Faces', reach_faces, 'Vertices', reachable_verts_phys, ...
          'FaceColor', reachable_color, 'FaceAlpha', reachable_alpha, ...
          'EdgeColor', 'none');
end

% --- 添加 3D 光影效果 (让曲面更有立体感) ---
% camlight('headlight');
% lighting(ax, 'gouraud');
% material(ax, 'dull'); 
% -------------------------------------------

% Draw hyperbolic constraint curves at theta = -pi, 0, +pi
x_range = [lb(1), lb(1) + shape(1)*eta(1)];
theta_vals = [-pi, 0, pi];
x1_line = linspace(x_range(1), x_range(2), 200);

% Draw hyperbolic constraint curves at theta = -pi, 0, +pi
theta_vals = [-pi, 0, pi];

% 计算交点边界 (精确值)
x1_intersect = sqrt(32/3); % 约等于 3.266
x2_intersect = sqrt(20/3); % 约等于 2.582

for theta = theta_vals
    
    % 1. 顶部弧线 (由约束2构成): 从左交点到右交点
    x1_top = linspace(-x1_intersect, x1_intersect, 100);
    x2_top = sqrt((x1_top.^2 + 16) / 4);
    
    % 2. 右侧弧线 (由约束1构成): 从上交点到下交点
    x2_right = linspace(x2_intersect, -x2_intersect, 100);
    x1_right = sqrt(x2_right.^2 + 4);
    
    % 3. 底部弧线 (由约束2构成): 从右交点到左交点
    x1_bottom = linspace(x1_intersect, -x1_intersect, 100);
    x2_bottom = -sqrt((x1_bottom.^2 + 16) / 4);
    
    % 4. 左侧弧线 (由约束1构成): 从下交点到上交点
    x2_left = linspace(-x2_intersect, x2_intersect, 100);
    x1_left = -sqrt(x2_left.^2 + 4);
    
    % 将四段首尾相接，拼成一个完整的闭合环
    x1_loop = [x1_top, x1_right, x1_bottom, x1_left];
    x2_loop = [x2_top, x2_right, x2_bottom, x2_left];
    
    % 绘制这条完美的闭合曲线
    plot3(ax, x1_loop, x2_loop, theta * ones(size(x1_loop)), 'r-', 'LineWidth', 3);
end

%% Formatting
y_range = [lb(2), lb(2) + shape(2)*eta(2)];
z_range = [lb(3), lb(3) + shape(3)*eta(3)];

axis(ax, [x_range, y_range, z_range]);
ax.Box = 'on';
view(ax, [azim, elev]);
grid(ax, 'on');

xlabel(ax, 'x', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(ax, 'y', 'FontSize', 14, 'FontWeight', 'bold');
zlabel(ax, '\theta', 'FontSize', 14, 'FontWeight', 'bold');

ax.FontSize = 12;
ax.FontWeight = 'bold';

title(ax, 'Candidate vs Reachable | Velocity: Projected', ...
      'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

% Legend
h_cand = patch(NaN, NaN, NaN, 'FaceColor', candidate_color, 'FaceAlpha', candidate_alpha);
h_reach = patch(NaN, NaN, NaN, 'FaceColor', reachable_color, 'FaceAlpha', reachable_alpha);
h_const = plot3(NaN, NaN, NaN, 'r-', 'LineWidth', 3);

leg = legend(ax, [h_cand, h_reach, h_const], ...
       {'Candidate', 'Reachable', 'Hyperbolic Constraint (\theta = -\pi, 0, +\pi)'}, ...
       'Location', 'northeast', 'FontSize', 12, 'Box', 'on');

try leg.BoxFaceAlpha = 0.95; catch; end
hold(ax, 'off');

%% Save plot
plots_dir = fullfile(project_root, 'plots');
if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end

output_path = fullfile(plots_dir, 'hyperbolic_demo_gpu_candidate_vs_reachable_projv_matlab.png');
exportgraphics(fig, output_path, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('Saved plot to: %s\n', output_path);