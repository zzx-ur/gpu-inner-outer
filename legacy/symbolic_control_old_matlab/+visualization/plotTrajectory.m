function h = plotTrajectory(trajectory, color, markerStyle)
    % plotTrajectory - 绘制状态轨迹
    %
    % 在三维空间中绘制系统轨迹。
    %
    % 用法:
    %   symctrl.visualization.plotTrajectory(trajectory)
    %   symctrl.visualization.plotTrajectory(trajectory, 'b', '-o')
    %
    % 输入:
    %   trajectory  - 轨迹数据，可以是:
    %                 - [3 x n] 矩阵
    %                 - 结构体，包含 .x 字段
    %   color       - 线条颜色 (默认 'b')
    %   markerStyle - 标记样式 (默认 '-o')
    %
    % 示例:
    %   traj = kinv.simulate(x0, 50);
    %   symctrl.visualization.plotTrajectory(traj, 'r', '-*');
    
    arguments
        trajectory
        color = 'b'
        markerStyle = '-o'
    end
    
    % 解析输入
    if isstruct(trajectory)
        X = trajectory.x;
    else
        X = trajectory;
    end
    
    % 绘制轨迹
    hold on;
    h = plot3(X(1, :), X(2, :), X(3, :), [color, markerStyle], 'LineWidth', 1.5);
    
    % 标记起点和终点（不出现在 legend 中）
    plot3(X(1, 1), X(2, 1), X(3, 1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'HandleVisibility', 'off');
    plot3(X(1, end), X(2, end), X(3, end), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
    
    xlabel('x_1');
    ylabel('x_2');
    zlabel('\theta');
    grid on;
    hold off;
end
