%% 独轮车模型 鲁棒K-Invariant Set 计算示例
% 该示例演示如何使用 symctrl 包计算独轮车模型的鲁棒K-不变集，
% 仅计算鲁棒可达集合，候选集与目标集合相同。
% 支持非线性状态约束。
%
% 作者: Symbolic Control Toolbox

clear;
close all;
clc;

%% ==================== 第一部分: 计算 K-Invariant Set ====================

%% 1. 系统参数定义
fprintf('=== 鲁棒 K-Invariant Set 计算 ===\n\n');

T = 1;  % 采样时间

% 控制约束 U = [0.2, 2] × [-1, 1]
bound_u = [0.2, 2; -1, 1];

% 扰动约束
bound_w = [-0.12, 0.12; -0.12, 0.12; -0.12, 0.12];

% 状态空间边界 (bounding box)
bound_x = [-3.5, 3.5; -3.5, 3.5; -pi, pi];

% 离散化步长
d_x = [0.1; 0.1; 0.1];

% 控制输入离散化点数
n_u = [3; 5];

% 状态约束 X: x1^2 - x2^2 <= 4, 4*x2^2 - x1^2 <= 16
stateConstraint = @(x) (x(1)^2 - x(2)^2 <= 4) && (4*x(2)^2 - x(1)^2 <= 16);

% 候选集区域
candidateRegion = [[-1; -1; -pi], [1; 1; pi]];

% 结果文件路径
kinvFile = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'kinv_unicycle_robust.mat');

%% 2. 计算 K-Invariant Set
fprintf('开始计算 (这可能需要几分钟时间)...\n\n');

sysModel = symctrl.SystemModel.createUnicycle(T, bound_w, bound_u);
stateSpace = symctrl.StateSpace(bound_x, d_x);
modes = symctrl.utils.generateModes(bound_u, n_u, true);

tic;
kinv = symctrl.KInvariant(sysModel, stateSpace, modes, candidateRegion, ...
                          'stateConstraint', stateConstraint, ...
                          'maxIter', 100, 'verbose', true);
computeTime = toc;
fprintf('\n计算完成! 耗时: %.2f 秒\n', computeTime);

kinv.save(kinvFile);

%% 3. 显示统计信息
stats = kinv.getStatistics();
fprintf('\n=== 统计信息 ===\n');
fprintf('总离散状态数: %d\n', stats.totalStates);
fprintf('候选集状态数: %d\n', stats.candidateStatesCount);
fprintf('满足约束的状态数: %d\n', stats.constrainedStatesCount);
fprintf('鲁棒可达状态数: %d\n', stats.reachableRobustCount);
fprintf('候选集覆盖率: %.1f%%\n', stats.candidateCoverage);
fprintf('迭代次数: %d\n', stats.iterations);
fprintf('收敛状态: %s\n', mat2str(stats.converged));

fprintf('\n=== 计算部分完成 ===\n');

