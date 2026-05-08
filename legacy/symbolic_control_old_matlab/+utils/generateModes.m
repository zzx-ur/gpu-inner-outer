function modes = generateModes(bound_u, n_u, sortByMagnitude)
    % generateModes - 生成离散控制模式
    %
    % 该函数根据控制边界和每个维度的离散点数生成所有控制模式组合。
    % 可选择按幅值排序，以优先尝试小输入。
    %
    % 用法:
    %   modes = symctrl.utils.generateModes(bound_u, n_u)
    %   modes = symctrl.utils.generateModes(bound_u, n_u, true)
    %
    % 输入:
    %   bound_u         - 控制边界 [n_u x 2]
    %   n_u             - 每个维度的离散点数 [n_u x 1]
    %   sortByMagnitude - 是否按幅值排序 (默认 true)
    %
    % 输出:
    %   modes - 控制模式矩阵 [dim_u x n_modes]
    %
    % 示例:
    %   bound_u = [0.25 1; -1 1];  % 速度和角速度范围
    %   n_u = [9; 11];             % 离散点数
    %   modes = symctrl.utils.generateModes(bound_u, n_u, true);
    
    arguments
        bound_u
        n_u
        sortByMagnitude = true
    end
    
    dim_u = size(bound_u, 1);
    
    % 为每个维度生成离散值
    modeValues = cell(dim_u, 1);
    for i = 1:dim_u
        modeValues{i} = linspace(bound_u(i, 1), bound_u(i, 2), n_u(i));
        
        % 如果需要按幅值排序
        if sortByMagnitude
            [~, sortIdx] = sort(abs(modeValues{i}));
            modeValues{i} = modeValues{i}(sortIdx);
        end
    end
    
    % 生成所有组合
    n_modes = prod(n_u);
    modes = zeros(dim_u, n_modes);
    
    idx = 1;
    if dim_u == 1
        modes = modeValues{1};
    elseif dim_u == 2
        for i = 1:n_u(1)
            for j = 1:n_u(2)
                modes(:, idx) = [modeValues{1}(i); modeValues{2}(j)];
                idx = idx + 1;
            end
        end
    else
        % 通用情况: 使用 ndgrid
        grids = cell(dim_u, 1);
        [grids{:}] = ndgrid(modeValues{:});
        for i = 1:dim_u
            modes(i, :) = grids{i}(:)';
        end
    end
end
