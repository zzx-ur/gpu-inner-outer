function xCorr = angleCorrection(x, angleDim)
    % angleCorrection - 角度校正函数
    %
    % 将角度维度归一化到 [-pi, pi] 范围内。
    %
    % 用法:
    %   xCorr = symctrl.utils.angleCorrection(x)
    %   xCorr = symctrl.utils.angleCorrection(x, 3)
    %
    % 输入:
    %   x        - 状态向量 [n x 1]
    %   angleDim - 角度维度索引 (默认为 3)
    %
    % 输出:
    %   xCorr - 校正后的状态向量
    %
    % 示例:
    %   x = [1; 2; 4];  % 第三维超出 [-pi, pi]
    %   xCorr = symctrl.utils.angleCorrection(x);
    %   % xCorr(3) ≈ 4 - 2*pi ≈ -2.28
    
    arguments
        x
        angleDim = 3
    end
    
    xCorr = x;
    
    % 将角度归一化到 [-pi, pi]
    while xCorr(angleDim) > pi
        xCorr(angleDim) = xCorr(angleDim) - 2*pi;
    end
    
    while xCorr(angleDim) < -pi
        xCorr(angleDim) = xCorr(angleDim) + 2*pi;
    end
end
