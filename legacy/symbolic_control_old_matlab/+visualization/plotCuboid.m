function h = plotCuboid(bbox, color, alpha)
    % plotCuboid - 绘制三维长方体
    %
    % 在当前图形中绘制一个半透明的三维长方体。
    %
    % 用法:
    %   h = symctrl.visualization.plotCuboid(bbox)
    %   h = symctrl.visualization.plotCuboid(bbox, 'r')
    %   h = symctrl.visualization.plotCuboid(bbox, 'r', 0.3)
    %
    % 输入:
    %   bbox  - 边界框 [3 x 2]，每行为 [min, max]
    %   color - 颜色 (默认 'b')
    %   alpha - 透明度 0-1 (默认 0.2)
    %
    % 输出:
    %   h - patch对象句柄
    %
    % 示例:
    %   bbox = [-1 1; -1 1; -pi pi];
    %   h = symctrl.visualization.plotCuboid(bbox, 'r', 0.3);
    
    arguments
        bbox
        color = 'b'
        alpha = 0.2
    end
    
    pmin = bbox(:, 1);
    pmax = bbox(:, 2);
    
    % 生成顶点
    x = [pmin(1), pmax(1)];
    y = [pmin(2), pmax(2)];
    z = [pmin(3), pmax(3)];
    [X, Y, Z] = meshgrid(x, y, z);
    
    % 创建面
    faces = [1 2 4 3; 1 2 6 5; 2 4 8 6; 3 4 8 7; 1 3 7 5; 5 6 8 7];
    vertices = [X(:), Y(:), Z(:)];
    
    % 绘制半透明面
    h = patch('Faces', faces, 'Vertices', vertices, ...
              'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', alpha);
    
    hold on;
    
    % 绘制边框线
    linewidth = 1;
    lineColor = 'k';
    
    edgeOpts = {lineColor, 'LineWidth', linewidth, 'HandleVisibility', 'off'};
    
    % x方向的线
    plot3([pmin(1), pmax(1)], [pmin(2), pmin(2)], [pmin(3), pmin(3)], edgeOpts{:});
    plot3([pmin(1), pmax(1)], [pmin(2), pmin(2)], [pmax(3), pmax(3)], edgeOpts{:});
    plot3([pmin(1), pmax(1)], [pmax(2), pmax(2)], [pmin(3), pmin(3)], edgeOpts{:});
    plot3([pmin(1), pmax(1)], [pmax(2), pmax(2)], [pmax(3), pmax(3)], edgeOpts{:});
    
    % y方向的线
    plot3([pmin(1), pmin(1)], [pmin(2), pmax(2)], [pmin(3), pmin(3)], edgeOpts{:});
    plot3([pmin(1), pmin(1)], [pmin(2), pmax(2)], [pmax(3), pmax(3)], edgeOpts{:});
    plot3([pmax(1), pmax(1)], [pmin(2), pmax(2)], [pmin(3), pmin(3)], edgeOpts{:});
    plot3([pmax(1), pmax(1)], [pmin(2), pmax(2)], [pmax(3), pmax(3)], edgeOpts{:});
    
    % z方向的线
    plot3([pmin(1), pmin(1)], [pmin(2), pmin(2)], [pmin(3), pmax(3)], edgeOpts{:});
    plot3([pmin(1), pmin(1)], [pmax(2), pmax(2)], [pmin(3), pmax(3)], edgeOpts{:});
    plot3([pmax(1), pmax(1)], [pmin(2), pmin(2)], [pmin(3), pmax(3)], edgeOpts{:});
    plot3([pmax(1), pmax(1)], [pmax(2), pmax(2)], [pmin(3), pmax(3)], edgeOpts{:});
    
end
