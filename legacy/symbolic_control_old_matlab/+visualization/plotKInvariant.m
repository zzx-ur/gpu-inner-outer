function plotKInvariant(kinv, options)
    % plotKInvariant - 可视化鲁棒K-不变集计算结果
    %
    % 绘制K-不变集的候选区域和状态空间边界。
    %
    % 用法:
    %   symctrl.visualization.plotKInvariant(kinv)
    %   symctrl.visualization.plotKInvariant(kinv, 'ShowBoundary', true)
    %
    % 输入:
    %   kinv    - KInvariant对象
    %   options - 可选参数:
    %       .ShowBoundary  - 是否显示边界 (默认 true)
    %       .CandidateColor- 候选区域颜色 (默认 'b')
    %       .BoundaryColor - 边界颜色 (默认 'k')
    %       .Alpha         - 透明度 (默认 0.2)
    %
    % 示例:
    %   kinv = symctrl.KInvariant(model, ss, modes, candidate);
    %   symctrl.visualization.plotKInvariant(kinv);
    
    arguments
        kinv
        options.ShowBoundary = true
        options.CandidateColor = 'b'
        options.BoundaryColor = 'k'
        options.Alpha = 0.2
    end
    
    import symctrl.visualization.plotCuboid
    
    figure;
    hold on;
    
    % 绘制候选区域
    h_candidate = plotCuboid(kinv.candidateRegion, options.CandidateColor, options.Alpha);
    
    % 绘制状态空间边界
    if options.ShowBoundary
        h_boundary = plotCuboid(kinv.stateSpace.bound, options.BoundaryColor, 0.05);
    end
    
    % 设置视角和轴
    view([30, 16]);
    axis equal;
    
    boundMargin = 0.5;
    ss = kinv.stateSpace;
    axis([ss.lb(1)-boundMargin, ss.ub(1)+boundMargin, ...
          ss.lb(2)-boundMargin, ss.ub(2)+boundMargin, ...
          ss.lb(3)-0.2, ss.ub(3)+0.2]);
    
    xlabel('x_1');
    ylabel('x_2');
    zlabel('\theta', 'Interpreter', 'latex');
    
    title(sprintf('鲁棒K-Invariant Set (迭代: %d, 收敛: %s)', ...
          kinv.iterations, mat2str(kinv.converged)));
    
    if options.ShowBoundary
        legend([h_candidate, h_boundary], ...
               {'候选集', '状态边界'}, 'Location', 'best');
    else
        legend(h_candidate, {'候选集'}, 'Location', 'best');
    end
    
    hold off;
end
