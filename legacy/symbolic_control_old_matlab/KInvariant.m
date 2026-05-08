classdef KInvariant
    % KInvariant - 鲁棒K-不变集计算类
    %
    % 该类计算鲁棒控制不变集(K-invariant set)。给定候选集合，
    % 迭代计算所有可以在有限步内到达候选集合的状态。
    %
    % 核心算法:
    %   1. 候选集合作为初始可达集合
    %   2. 迭代扩展：找到所有可以一步到达当前可达集合的状态
    %   3. 收敛条件：当候选集合是其可达集合的子集时停止
    %
    % 支持功能:
    %   - 鲁棒可达集计算(考虑扰动)
    %   - 非线性状态约束
    %
    % 用法:
    %   kinv = symctrl.KInvariant(sysModel, stateSpace, modes, candidateRegion)
    %   kinv = symctrl.KInvariant(..., 'stateConstraint', @(x) x(1)^2 + x(2)^2 <= 1)
    %
    % 示例:
    %   model = symctrl.SystemModel.createUnicycle(1, bound_w, bound_u);
    %   ss = symctrl.StateSpace([-3 3; -3 3; -pi pi], [0.1; 0.1; 0.2]);
    %   modes = [0.5 0.5; 0 0.1];  % 两个控制模式
    %   candidate = [[-1;-1;-pi], [1;1;pi]];
    %   constraint = @(x) x(1)^2 - x(2)^2 <= 4;
    %   kinv = symctrl.KInvariant(model, ss, modes, candidate, 'stateConstraint', constraint);
    
    properties (SetAccess = private)
        % 系统模型
        sysModel
        
        % 状态空间
        stateSpace
        
        % 控制模式矩阵 [n_u x n_modes]
        modes
        
        % 候选集区域 [n_dim x 2]
        candidateRegion
        
        % 候选集内的所有状态索引
        candidateStates
        
        % 状态约束函数 @(x) -> boolean
        stateConstraint
        
        % 满足状态约束的状态索引
        constrainedStates
        
        % 鲁棒可达集值 (到达候选集的步数)
        reachSetRobust
        
        % 控制器映射 (状态 -> 模式索引)
        controllers
        
        % 收敛时的迭代步数
        iterations
        
        % 是否收敛
        converged
        
        % 收敛消息
        message
    end
    
    methods
        function obj = KInvariant(sysModel, stateSpace, modes, candidateRegion, options)
            % KInvariant 构造函数
            %
            % 输入:
            %   sysModel        - SystemModel对象
            %   stateSpace      - StateSpace对象
            %   modes           - 控制模式矩阵 [n_u x n_modes]
            %   candidateRegion - 候选集区域 [n_dim x 2]
            %   options         - 可选参数结构体
            %       .stateConstraint - 状态约束函数 @(x) -> boolean (默认无约束)
            %       .maxIter    - 最大迭代次数 (默认 1000)
            %       .verbose    - 是否显示进度 (默认 true)
            
            arguments
                sysModel
                stateSpace
                modes
                candidateRegion
                options.stateConstraint = []
                options.maxIter = 1000
                options.verbose = true
            end
            
            obj.sysModel = sysModel;
            obj.stateSpace = stateSpace;
            obj.modes = modes;
            obj.candidateRegion = candidateRegion;
            obj.stateConstraint = options.stateConstraint;
            
            % 计算候选集的状态索引
            obj.candidateStates = stateSpace.regionToStates(candidateRegion);
            
            % 计算满足状态约束的状态
            obj.constrainedStates = obj.computeConstrainedStates(options.verbose);
            
            % 执行K-不变集计算
            [obj.reachSetRobust, obj.controllers, ...
             obj.iterations, obj.converged, obj.message] = ...
                obj.computeKInvariant(options.maxIter, options.verbose);
        end
        
        function constrainedStates = computeConstrainedStates(obj, verbose)
            % 计算满足状态约束的状态
            %
            % 输出:
            %   constrainedStates - 满足约束的状态索引向量
            
            if isempty(obj.stateConstraint)
                % 无约束，所有状态都满足
                constrainedStates = (1:obj.stateSpace.n_states)';
                return;
            end
            
            if verbose
                fprintf('计算满足状态约束的状态...\n');
            end
            
            n_states = obj.stateSpace.n_states;
            validMask = false(n_states, 1);
            
            for q = 1:n_states
                x = obj.stateSpace.stateToCoord(q);
                if obj.stateConstraint(x)
                    validMask(q) = true;
                end
            end
            
            constrainedStates = find(validMask);
            
            if verbose
                fprintf('  - 满足约束的状态数: %d/%d (%.1f%%)\n', ...
                    length(constrainedStates), n_states, 100*length(constrainedStates)/n_states);
            end
        end
        
        function [reachRobust, ctrl, iter, success, msg] = ...
                computeKInvariant(obj, maxIter, verbose)
            % 计算鲁棒K-不变集
            %
            % 算法流程:
            %   1. 候选集 K 作为目标集
            %   2. K_1 = 所有能一步到达 K 的状态
            %   3. K_2 = 所有能一步到达 K ∪ K_1 的状态
            %   4. 重复直到 K ⊆ K_1 ∪ K_2 ∪ ... ∪ K_p
            %
            % 返回:
            %   reachRobust  - 到达目标的步数 (inf表示不可达)
            %   ctrl         - 控制器映射
            %   iter         - 迭代步数
            %   success      - 是否收敛
            %   msg          - 状态消息
            
            n_states = obj.stateSpace.n_states;
            n_modes = size(obj.modes, 2);
            
            % 预计算所有状态的转移关系
            if verbose
                fprintf('正在构建符号模型 (共 %d 个状态)...\n', n_states);
            end
            [DeltaRobust, validStates] = obj.buildTransitionModel(n_states, n_modes, verbose);
            
            % 检查候选集内的状态是否都有有效转移
            if any(~validStates(obj.candidateStates))
                warning('部分候选集状态没有有效转移');
            end
            
            % 初始化
            reachRobust = inf(n_states, 1);  % 所有状态初始为不可达
            ctrl = zeros(n_states, 1);
            
            % 标记候选集 K 为目标集 (用 0 表示目标)
            isTarget = false(n_states, 1);
            isTarget(obj.candidateStates) = true;
            
            % 迭代计算
            if verbose
                fprintf('开始迭代计算鲁棒K-不变集...\n');
            end
            
            success = false;
            msg = '';
            
            for iter = 1:maxIter
                reachRobustPrev = reachRobust;
                newReachable = 0;
                
                for q = 1:n_states
                    % 跳过已经可达的状态
                    if reachRobust(q) ~= inf
                        continue;
                    end
                    
                    for p = 1:n_modes
                        idx = (q-1)*n_modes + p;
                        DelRobust = DeltaRobust(idx, :);
                        
                        % 检查转移是否有效
                        if ~all(DelRobust)
                            continue;
                        end
                        
                        % 获取后继状态集合
                        Q_succ = obj.getSuccessorStates(DelRobust(1), DelRobust(2));
                        
                        % 检查所有后继状态是否都在目标集或已可达
                        % iter=1时: 后继必须在 K 内 (isTarget)
                        % iter>1时: 后继必须在 K 内或 reachRobust <= iter-1
                        allSuccReachable = true;
                        for qs = Q_succ
                            if ~isTarget(qs) && reachRobust(qs) > iter - 1
                                allSuccReachable = false;
                                break;
                            end
                        end
                        
                        if allSuccReachable
                            reachRobust(q) = iter;
                            ctrl(q) = p;
                            newReachable = newReachable + 1;
                            break;  % 找到有效控制即可
                        end
                    end
                end
                
                % 检查收敛条件: K ⊆ K_1 ∪ K_2 ∪ ... ∪ K_iter
                candidateCovered = all(reachRobust(obj.candidateStates) ~= inf);
                
                if candidateCovered
                    success = true;
                    msg = sprintf('收敛成功! 迭代%d步，候选集是鲁棒K-不变集', iter);
                    if verbose
                        fprintf('%s\n', msg);
                    end
                    break;
                end
                
                % 检查是否停滞 (没有新状态被标记为可达)
                if newReachable == 0
                    msg = '鲁棒可达集停止扩展，候选集无法完全覆盖';
                    if verbose
                        fprintf('警告: %s\n', msg);
                    end
                    break;
                end
                
                if verbose
                    nReachable = sum(reachRobust(obj.candidateStates) ~= inf);
                    nTotal = length(obj.candidateStates);
                    fprintf('迭代 %d: 新增 %d 个可达状态, 候选集覆盖率 %.1f%% (%d/%d)\n', ...
                        iter, newReachable, 100*nReachable/nTotal, nReachable, nTotal);
                end
            end
            
            if iter >= maxIter && ~success
                msg = sprintf('达到最大迭代次数 %d', maxIter);
                if verbose
                    fprintf('警告: %s\n', msg);
                end
            end
            
            % 为已可达的目标集状态设置特殊标记 (0表示在目标集内且可达)
            for q = obj.candidateStates'
                if reachRobust(q) ~= inf
                    % 保持 reachRobust 值表示到达目标的步数
                    % 但对于目标集内的状态，如果它可达，意味着它可以回到目标集
                end
            end
        end
        
        function [DeltaRobust, validStates] = buildTransitionModel(obj, n_states, n_modes, verbose)
            % 构建状态转移模型 (仅鲁棒)
            %
            % 返回:
            %   DeltaRobust  - 鲁棒转移表 [n_states*n_modes x 2]
            %   validStates  - 有效状态标记
            
            if nargin < 4
                verbose = false;
            end
            
            DeltaRobust = zeros(n_states * n_modes, 2);
            validStates = false(n_states, 1);
            
            ss = obj.stateSpace;
            model = obj.sysModel;
            d_x = ss.d_x;
            
            % 进度显示间隔
            progressInterval = max(1, floor(n_states / 20));
            
            for q = 1:n_states
                % 显示进度
                if verbose && (mod(q, progressInterval) == 0 || q == n_states)
                    fprintf('  构建进度: %d/%d (%.1f%%)\n', q, n_states, 100*q/n_states);
                end
                
                x_c = ss.stateToCoord(q);
                
                % 检查当前状态约束
                if ~isempty(obj.stateConstraint) && ~obj.stateConstraint(x_c)
                    continue;
                end
                
                for p = 1:n_modes
                    u = obj.modes(:, p);
                    idx = (q-1)*n_modes + p;
                    
                    % 计算鲁棒可达集
                    [x_min_r, x_max_r] = model.reachSetWithDisturbance(x_c, u, d_x);
                    x_min_r = ss.correctAngle(x_min_r);
                    x_max_r = ss.correctAngle(x_max_r);
                    
                    if ss.isInBound([x_min_r, x_max_r])
                        min_succ = ss.coordToState(x_min_r, 0);
                        max_succ = ss.coordToState(x_max_r, 1);
                        
                        % 检查所有后继状态是否满足约束
                        if ~isempty(obj.stateConstraint)
                            Q_succ = obj.getSuccessorStates(min_succ, max_succ);
                            validTransition = true;
                            for qs = Q_succ
                                xs = ss.stateToCoord(qs);
                                if ~obj.stateConstraint(xs)
                                    validTransition = false;
                                    break;
                                end
                            end
                            if ~validTransition
                                continue;
                            end
                        end
                        
                        DeltaRobust(idx, :) = [min_succ, max_succ];
                        validStates(q) = true;
                    end
                end
            end
        end
        
        function Q_succ = getSuccessorStates(obj, minState, maxState)
            % 获取状态范围内的所有后继状态
            %
            % 输入:
            %   minState - 最小状态索引
            %   maxState - 最大状态索引
            % 输出:
            %   Q_succ - 后继状态集合
            
            ss = obj.stateSpace;
            Imin = ss.stateToIndex(minState);
            Imax = ss.stateToIndex(maxState);
            
            % 处理角度环绕情况
            if Imin(ss.n_dim) <= Imax(ss.n_dim)
                Q_succ = ss.indexBoxToStates(Imin, Imax)';
            else
                % 角度跨越边界，需要分成两部分
                Imin_wrap = Imin; Imin_wrap(ss.n_dim) = 1;
                Imax_wrap = Imax; Imax_wrap(ss.n_dim) = ss.n_x(ss.n_dim);
                Q_succ = ss.indexBoxToStates(Imin_wrap, Imax_wrap)';
            end
        end
        
        function u = getControl(obj, x)
            % 获取给定状态的控制输入
            %
            % 输入:
            %   x - 连续状态
            % 输出:
            %   u - 控制输入，如果状态不可达返回空
            
            q = obj.stateSpace.coordToState(x, 0);
            modeIdx = obj.controllers(q);
            
            if modeIdx == 0 || obj.reachSetRobust(q) == inf
                u = [];
                warning('状态 [%.2f, %.2f, %.2f] 不在鲁棒可达集内', x(1), x(2), x(3));
            else
                u = obj.modes(:, modeIdx);
            end
        end
        
        function stepsToTarget = getStepsToTarget(obj, x)
            % 获取从给定状态到目标的步数
            %
            % 输入:
            %   x - 连续状态
            % 输出:
            %   stepsToTarget - 到目标的步数，inf表示不可达
            
            q = obj.stateSpace.coordToState(x, 0);
            stepsToTarget = obj.reachSetRobust(q);
        end
        
        function trajectory = simulate(obj, x0, maxSteps, useDisturbance, distFunc)
            % 从初始状态模拟轨迹
            %
            % 输入:
            %   x0             - 初始状态
            %   maxSteps       - 最大步数 (默认 100)
            %   useDisturbance - 是否添加扰动 (默认 false)
            %   distFunc       - 扰动生成函数 (默认均匀随机)
            % 输出:
            %   trajectory - 结构体包含:
            %       .x - 状态轨迹 [n_dim x nSteps]
            %       .u - 控制轨迹 [n_u x nSteps-1]
            %       .success - 是否到达目标
            
            arguments
                obj
                x0
                maxSteps = 100
                useDisturbance = false
                distFunc = []
            end
            
            if isempty(distFunc)
                % 默认: 边界上的均匀随机扰动
                distFunc = @() obj.sysModel.bound_w(:,1) + ...
                    (obj.sysModel.bound_w(:,2) - obj.sysModel.bound_w(:,1)) .* rand(obj.sysModel.n_w, 1);
            end
            
            trajectory.x = x0;
            trajectory.u = [];
            trajectory.success = false;
            
            x = x0;
            
            for step = 1:maxSteps
                % 获取控制
                u = obj.getControl(x);
                if isempty(u)
                    warning('轨迹在步骤 %d 中断: 状态不可达', step);
                    break;
                end
                
                % 检查是否到达目标
                if obj.getStepsToTarget(x) == 0
                    trajectory.success = true;
                    break;
                end
                
                % 执行状态转移
                if useDisturbance
                    w = distFunc();
                    x_next = obj.sysModel.stepWithDisturbance(x, u, w);
                else
                    x_next = obj.sysModel.stepNominal(x, u);
                end
                
                x_next = obj.stateSpace.correctAngle(x_next);
                
                trajectory.x = [trajectory.x, x_next];
                trajectory.u = [trajectory.u, u];
                
                x = x_next;
            end
        end
        
        function stats = getStatistics(obj)
            % 获取鲁棒K-不变集的统计信息
            %
            % 输出:
            %   stats - 结构体包含统计信息
            
            stats.totalStates = obj.stateSpace.n_states;
            stats.candidateStatesCount = length(obj.candidateStates);
            stats.constrainedStatesCount = length(obj.constrainedStates);
            stats.reachableRobustCount = sum(obj.reachSetRobust ~= inf);
            stats.candidateCoverage = sum(obj.reachSetRobust(obj.candidateStates) ~= inf) / ...
                                       length(obj.candidateStates) * 100;
            stats.iterations = obj.iterations;
            stats.converged = obj.converged;
        end
        
        function save(obj, filepath)
            % 将KInvariant对象保存到文件
            %
            % 用法:
            %   kinv.save('results/kinv_unicycle_robust.mat')
            %
            % 输入:
            %   filepath - 保存路径，建议使用 .mat 扩展名
            
            % 确保目标目录存在
            [dirPath, ~, ~] = fileparts(filepath);
            if ~isempty(dirPath) && ~exist(dirPath, 'dir')
                mkdir(dirPath);
            end
            
            kinv = obj; %#ok<NASGU>
            save(filepath, 'kinv', '-v7.3');
            fprintf('鲁棒K-Invariant 已保存至: %s\n', filepath);
        end
    end
    
    methods (Static)
        function obj = load(filepath)
            % 从文件加载KInvariant对象
            %
            % 用法:
            %   kinv = symctrl.KInvariant.load('results/kinv_unicycle_robust.mat')
            %
            % 输入:
            %   filepath - 文件路径
            % 输出:
            %   obj - 加载的KInvariant对象
            
            if ~exist(filepath, 'file')
                error('文件不存在: %s', filepath);
            end
            
            data = load(filepath, 'kinv');
            obj = data.kinv;
            fprintf('鲁棒K-Invariant 已从文件加载: %s\n', filepath);
        end
    end
end
