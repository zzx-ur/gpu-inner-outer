classdef StateSpace
    % StateSpace - 状态空间离散化类
    %
    % 该类管理状态空间的离散化，提供连续状态与离散状态索引之间的转换功能。
    %
    % 用法:
    %   ss = symctrl.StateSpace(bound, d_x)
    %
    % 输入参数:
    %   bound - 状态空间边界 [n_x x 2]，每行为 [lb, ub]
    %   d_x   - 各维度离散化步长 [n_x x 1]
    %
    % 示例:
    %   bound = [-3 3; -3 3; -pi pi];  % 三维状态空间
    %   d_x = [0.1; 0.1; 0.2];         % 离散化步长
    %   ss = symctrl.StateSpace(bound, d_x);
    
    properties (SetAccess = private)
        % 状态空间边界 [n_x x 2]
        bound
        
        % 离散化步长 [n_x x 1]
        d_x
        
        % 状态维度
        n_dim
        
        % 每个维度的离散点数 [n_x x 1]
        n_x
        
        % 用于状态索引计算的累积乘积向量 [n_x+1 x 1]
        p_x
        
        % 总离散状态数
        n_states
        
        % 下界向量
        lb
        
        % 上界向量
        ub
    end
    
    methods
        function obj = StateSpace(bound, d_x)
            % StateSpace 构造函数
            %
            % 输入:
            %   bound - 状态空间边界 [n_x x 2]
            %   d_x   - 离散化步长 [n_x x 1]
            
            obj.bound = bound;
            obj.d_x = d_x;
            obj.n_dim = size(bound, 1);
            obj.lb = bound(:, 1);
            obj.ub = bound(:, 2);
            
            % 计算每个维度的离散点数
            obj.n_x = ceil((obj.ub - obj.lb) ./ d_x);
            
            % 计算状态索引的累积乘积向量
            obj.p_x = [1; cumprod(obj.n_x)];
            
            % 总状态数
            obj.n_states = obj.p_x(end);
        end
        
        function x = stateToCoord(obj, q)
            % 离散状态索引转换为连续坐标(单元中心)
            %
            % 输入:
            %   q - 离散状态索引
            % 输出:
            %   x - 连续坐标(单元中心)
            
            I = obj.stateToIndex(q);
            x = obj.indexToCoord(I);
        end
        
        function q = coordToState(obj, x, roundMode)
            % 连续坐标转换为离散状态索引
            %
            % 输入:
            %   x         - 连续坐标
            %   roundMode - 取整模式: 0=floor, 1=ceil
            % 输出:
            %   q - 离散状态索引
            
            if nargin < 3
                roundMode = 0;
            end
            
            I = obj.coordToIndex(x, roundMode);
            q = obj.indexToState(I);
        end
        
        function I = stateToIndex(obj, q)
            % 离散状态索引转换为多维索引
            %
            % 输入:
            %   q - 离散状态索引 (标量)
            % 输出:
            %   I - 多维索引 [n_dim x 1]
            
            I = zeros(obj.n_dim, 1);
            for k = 1:obj.n_dim
                I(k) = floor(mod(q-1, obj.p_x(k+1)) / obj.p_x(k)) + 1;
            end
        end
        
        function q = indexToState(obj, I)
            % 多维索引转换为离散状态索引
            %
            % 输入:
            %   I - 多维索引 [n_dim x 1]
            % 输出:
            %   q - 离散状态索引 (标量)
            
            q = (I - 1)' * obj.p_x(1:obj.n_dim) + 1;
        end
        
        function x = indexToCoord(obj, I)
            % 多维索引转换为连续坐标(单元中心)
            %
            % 输入:
            %   I - 多维索引 [n_dim x 1]
            % 输出:
            %   x - 连续坐标(单元中心)
            
            x = obj.lb + (I - 0.5) .* obj.d_x;
        end
        
        function I = coordToIndex(obj, x, roundMode)
            % 连续坐标转换为多维索引
            %
            % 输入:
            %   x         - 连续坐标
            %   roundMode - 取整模式: 0=floor, 1=ceil
            % 输出:
            %   I - 多维索引 [n_dim x 1]
            
            if roundMode == 0
                I = floor((x - obj.lb) ./ obj.d_x) + 1;
            else
                I = ceil((x - obj.lb) ./ obj.d_x);
            end
        end
        
        function states = regionToStates(obj, region)
            % 将一个矩形区域转换为其包含的所有状态索引
            %
            % 输入:
            %   region - 区域边界 [n_dim x 2]
            % 输出:
            %   states - 区域内所有状态索引的列向量
            
            Imin = obj.coordToIndex(region(:,1), 0);
            Imax = obj.coordToIndex(region(:,2), 1);
            states = obj.indexBoxToStates(Imin, Imax);
        end
        
        function states = indexBoxToStates(obj, Imin, Imax)
            % 将索引范围盒转换为所有状态索引
            %
            % 输入:
            %   Imin - 最小多维索引 [n_dim x 1]
            %   Imax - 最大多维索引 [n_dim x 1]
            % 输出:
            %   states - 盒内所有状态索引的列向量
            
            L = 0;
            for k = obj.n_dim:-1:1
                Lp = [];
                for j = L'
                    newStates = ((Imin(k):Imax(k))' - 1) * obj.p_x(k) + j;
                    Lp = [Lp; newStates];
                end
                L = Lp;
            end
            states = L + 1;
        end
        
        function valid = isInBound(obj, x)
            % 检查坐标是否在状态空间边界内
            %
            % 输入:
            %   x - 连续坐标 [n_dim x 1] 或 [n_dim x 2] (区域)
            % 输出:
            %   valid - 逻辑值，是否在边界内
            
            if size(x, 2) == 1
                valid = all(x >= obj.lb) && all(x <= obj.ub);
            else
                valid = all(x(:,1) >= obj.lb) && all(x(:,2) <= obj.ub);
            end
        end
        
        function xCorr = correctAngle(obj, x, angleDim)
            % 角度校正 - 将角度维度限制在 [-pi, pi] 范围内
            %
            % 输入:
            %   x        - 状态向量
            %   angleDim - 角度维度索引 (默认为最后一个维度)
            % 输出:
            %   xCorr - 校正后的状态向量
            
            if nargin < 3
                angleDim = obj.n_dim;
            end
            
            xCorr = x;
            while xCorr(angleDim) > pi
                xCorr(angleDim) = xCorr(angleDim) - 2*pi;
            end
            while xCorr(angleDim) < -pi
                xCorr(angleDim) = xCorr(angleDim) + 2*pi;
            end
        end
    end
end
