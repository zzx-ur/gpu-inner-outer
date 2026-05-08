classdef SystemModel
    % SystemModel - 定义系统动力学模型
    %
    % 该类封装了带扰动和无扰动的系统动力学模型，包括状态转移函数和
    % 用于可达集过逼近的Jacobian矩阵边界。
    %
    % 用法:
    %   model = symctrl.SystemModel(f, F, Jf_x, Jf_w, JF_x, bound_w, bound_u)
    %
    % 输入参数:
    %   f      - 带扰动的系统动力学函数 f(x, u, w)
    %   F      - 无扰动的系统动力学函数 F(x, u)
    %   Jf_x   - f对x的Jacobian边界函数 Jf_x(u)
    %   Jf_w   - f对w的Jacobian边界函数 Jf_w(u)
    %   JF_x   - F对x的Jacobian边界函数 JF_x(u)
    %   bound_w- 扰动边界 [n_w x 2] 矩阵
    %   bound_u- 控制边界 [n_u x 2] 矩阵
    %
    % 示例:
    %   % 独轮车模型
    %   T = 1;  % 采样时间
    %   f = @(x,u,w) [x(1)+T*u(1)*cos(x(3))+T*w(1);...
    %                 x(2)+T*u(1)*sin(x(3))+T*w(2);...
    %                 x(3)+T*u(2)+T*w(3)];
    %   F = @(x,u) [x(1)+T*u(1)*cos(x(3));...
    %               x(2)+T*u(1)*sin(x(3));...
    %               x(3)+T*u(2)];
    %   model = symctrl.SystemModel(f, F, Jf_x, Jf_w, JF_x, bound_w, bound_u);
    
    properties (SetAccess = private)
        % 带扰动的系统动力学 x_next = f(x, u, w)
        f
        
        % 无扰动的系统动力学 x_next = F(x, u)
        F
        
        % f对状态x的Jacobian矩阵边界 (用于可达集过逼近)
        Jf_x
        
        % f对扰动w的Jacobian矩阵边界
        Jf_w
        
        % F对状态x的Jacobian矩阵边界
        JF_x
        
        % 扰动边界 [n_w x 2]: [lb, ub]
        bound_w
        
        % 控制输入边界 [n_u x 2]: [lb, ub]
        bound_u
        
        % 扰动维度
        n_w
        
        % 控制输入维度
        n_u
        
        % 扰动范围 (ub - lb)
        d_w
        
        % 扰动中心值
        w_c
    end
    
    methods
        function obj = SystemModel(f, F, Jf_x, Jf_w, JF_x, bound_w, bound_u)
            % SystemModel 构造函数
            %
            % 输入:
            %   f       - 带扰动的系统动力学函数句柄 f(x, u, w)
            %   F       - 无扰动的系统动力学函数句柄 F(x, u)
            %   Jf_x    - Jacobian边界函数 Jf_x(u)
            %   Jf_w    - Jacobian边界函数 Jf_w(u)
            %   JF_x    - Jacobian边界函数 JF_x(u)
            %   bound_w - 扰动边界 [n_w x 2]
            %   bound_u - 控制边界 [n_u x 2]
            
            obj.f = f;
            obj.F = F;
            obj.Jf_x = Jf_x;
            obj.Jf_w = Jf_w;
            obj.JF_x = JF_x;
            obj.bound_w = bound_w;
            obj.bound_u = bound_u;
            
            % 计算扰动相关参数
            obj.n_w = size(bound_w, 1);
            obj.n_u = size(bound_u, 1);
            obj.d_w = bound_w(:,2) - bound_w(:,1);
            obj.w_c = 0.5 * (bound_w(:,1) + bound_w(:,2));
        end
        
        function x_next = stepWithDisturbance(obj, x, u, w)
            % 带扰动的状态转移
            %
            % 输入:
            %   x - 当前状态
            %   u - 控制输入
            %   w - 扰动
            % 输出:
            %   x_next - 下一状态
            
            x_next = obj.f(x, u, w);
        end
        
        function x_next = stepNominal(obj, x, u)
            % 无扰动的状态转移(标称模型)
            %
            % 输入:
            %   x - 当前状态
            %   u - 控制输入
            % 输出:
            %   x_next - 下一状态
            
            x_next = obj.F(x, u);
        end
        
        function [x_succ_min, x_succ_max] = reachSetWithDisturbance(obj, x_c, u, d_x)
            % 计算带扰动的可达集过逼近
            %
            % 输入:
            %   x_c - 状态单元中心
            %   u   - 控制输入
            %   d_x - 状态空间离散化步长
            % 输出:
            %   x_succ_min - 可达集最小边界
            %   x_succ_max - 可达集最大边界
            
            x_succ_c = obj.f(x_c, u, obj.w_c);
            d_x_succ = obj.Jf_x(u) * d_x * 0.5 + obj.Jf_w(u) * obj.d_w * 0.5;
            x_succ_min = x_succ_c - d_x_succ;
            x_succ_max = x_succ_c + d_x_succ;
        end
        
        function [x_succ_min, x_succ_max] = reachSetNominal(obj, x_c, u, d_x)
            % 计算无扰动的可达集过逼近
            %
            % 输入:
            %   x_c - 状态单元中心
            %   u   - 控制输入
            %   d_x - 状态空间离散化步长
            % 输出:
            %   x_succ_min - 可达集最小边界
            %   x_succ_max - 可达集最大边界
            
            x_succ_c = obj.F(x_c, u);
            d_x_succ = obj.JF_x(u) * d_x * 0.5;
            x_succ_min = x_succ_c - d_x_succ;
            x_succ_max = x_succ_c + d_x_succ;
        end
    end
    
    methods (Static)
        function model = createUnicycle(T, bound_w, bound_u)
            % 创建独轮车模型
            %
            % 输入:
            %   T       - 采样时间
            %   bound_w - 扰动边界 [3 x 2]
            %   bound_u - 控制边界 [2 x 2]
            % 输出:
            %   model   - SystemModel对象
            
            % 带扰动的动力学
            f = @(x,u,w) [x(1) + T*u(1)*cos(x(3)) + T*w(1);...
                          x(2) + T*u(1)*sin(x(3)) + T*w(2);...
                          x(3) + T*u(2) + T*w(3)];
            
            % 无扰动的动力学
            F = @(x,u) [x(1) + T*u(1)*cos(x(3));...
                        x(2) + T*u(1)*sin(x(3));...
                        x(3) + T*u(2)];
            
            % Jacobian边界
            Jf_x = @(u) [1 0 T*abs(u(1));...
                         0 1 T*abs(u(1));...
                         0 0 1];
            
            Jf_w = @(u) [T 0 0;...
                         0 T 0;...
                         0 0 T];
            
            JF_x = @(u) [1 0 T*abs(u(1));...
                         0 1 T*abs(u(1));...
                         0 0 1];
            
            model = symctrl.SystemModel(f, F, Jf_x, Jf_w, JF_x, bound_w, bound_u);
        end
    end
end
