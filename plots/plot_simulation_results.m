%% Visualize simulation_results.mat - State trajectories with constraints
%
% This script loads simulation results and plots the first two state dimensions
% (x, y) along with the hyperbolic constraint boundary and candidate set.

clear; close all; clc;

%% Configuration
% Determine project root
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

% Load configuration from hyperbolic_case.cfg
config_file = fullfile(project_root, 'examples', 'hyperbolic_case.cfg');
config = parse_config(config_file);

% Extract parameters
hyperbolic_a = config.hyperbolic_a;
hyperbolic_b = config.hyperbolic_b;
hyperbolic_c = config.hyperbolic_c;

candidate_lb = config.candidate_lb;
candidate_ub = config.candidate_ub;

state_lb = config.state_lb;
state_ub = config.state_ub;

%% Load simulation results
result_file = fullfile(project_root, 'plots', 'simulation_results.mat');
fprintf('Loading %s...\n', result_file);

data = load(result_file);

% Extract trajectory data - collect all trajectory_* fields
trajectories = {};
field_names = fieldnames(data);
traj_count = 0;

for i = 1:length(field_names)
    field_name = field_names{i};
    if startsWith(field_name, 'trajectory_')
        traj_count = traj_count + 1;
        trajectories{traj_count} = data.(field_name);
    end
end

if traj_count == 0
    fprintf('Available variables in .mat file:\n');
    disp(field_names);
    error('Could not find trajectory data. Check variable names above.');
end

fprintf('Found %d trajectories\n', traj_count);

%% Create figure
fig = figure('Color', 'white', 'Position', [100 100 1000 800]);
ax = axes('Parent', fig);
hold(ax, 'on');

%% Plot trajectories (first two dimensions: x, y)
if iscell(trajectories)
    % Multiple trajectories
    num_trajectories = length(trajectories);
    colors = parula(num_trajectories);
    
    for i = 1:num_trajectories
        traj = trajectories{i};
        if size(traj, 2) >= 2
            plot(ax, traj(:, 1), traj(:, 2), '-', 'Color', colors(i, :), ...
                 'LineWidth', 1.5, 'DisplayName', sprintf('Trajectory %d', i));
        end
    end
else
    % Single trajectory or matrix of trajectories
    if size(trajectories, 2) >= 2
        plot(ax, trajectories(:, 1), trajectories(:, 2), 'b-', 'LineWidth', 2, ...
             'DisplayName', 'Trajectory');
    end
end

%% Plot candidate set (rectangle in x-y plane)
cand_x_min = candidate_lb(1);
cand_x_max = candidate_ub(1);
cand_y_min = candidate_lb(2);
cand_y_max = candidate_ub(2);

% Draw candidate set rectangle
rect_x = [cand_x_min, cand_x_max, cand_x_max, cand_x_min, cand_x_min];
rect_y = [cand_y_min, cand_y_min, cand_y_max, cand_y_max, cand_y_min];
fill(ax, rect_x, rect_y, [0.2, 0.2, 0.6], 'FaceAlpha', 0.2, 'EdgeColor', 'b', ...
     'LineWidth', 2, 'DisplayName', 'Candidate Set');

%% Plot hyperbolic constraint boundary
% Constraint 1: x1^2 - x2^2 <= a  =>  x2 = ±sqrt(x1^2 - a)
% Constraint 2: b*x2^2 - x1^2 <= c  =>  x2 = ±sqrt((x1^2 + c) / b)

x_range = [state_lb(1), state_ub(1)];
x_line = linspace(x_range(1), x_range(2), 500);

% Constraint 1: x1^2 - x2^2 = a
% Valid for |x1| >= sqrt(a)
x1_threshold = sqrt(hyperbolic_a);

% Left branch: x1 <= -sqrt(a)
x1_left = x_line(x_line <= -x1_threshold);
if ~isempty(x1_left)
    x2_pos_left = sqrt(x1_left.^2 - hyperbolic_a);
    x2_neg_left = -x2_pos_left;
    plot(ax, x1_left, x2_pos_left, 'r-', 'LineWidth', 2.5);
    plot(ax, x1_left, x2_neg_left, 'r-', 'LineWidth', 2.5);
end

% Right branch: x1 >= sqrt(a)
x1_right = x_line(x_line >= x1_threshold);
if ~isempty(x1_right)
    x2_pos_right = sqrt(x1_right.^2 - hyperbolic_a);
    x2_neg_right = -x2_pos_right;
    plot(ax, x1_right, x2_pos_right, 'r-', 'LineWidth', 2.5);
    plot(ax, x1_right, x2_neg_right, 'r-', 'LineWidth', 2.5);
end

% Constraint 2: b*x2^2 - x1^2 = c  =>  x2 = ±sqrt((x1^2 + c) / b)
x2_pos_c2 = sqrt((x_line.^2 + hyperbolic_c) / hyperbolic_b);
x2_neg_c2 = -x2_pos_c2;
plot(ax, x_line, x2_pos_c2, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Hyperbolic Constraint');
plot(ax, x_line, x2_neg_c2, 'r-', 'LineWidth', 2.5);

%% Formatting
ax.Box = 'on';
grid(ax, 'on');
ax.GridLineStyle = '--';
ax.GridAlpha = 0.3;

xlabel(ax, 'x_1 (x position)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(ax, 'x_2 (y position)', 'FontSize', 14, 'FontWeight', 'bold');

ax.FontSize = 12;
ax.FontWeight = 'bold';

title(ax, 'Simulation Results: State Trajectories with Hyperbolic Constraints', ...
      'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

% Set axis limits
x_margin = 0.2;
y_margin = 0.2;
xlim(ax, [state_lb(1) - x_margin, state_ub(1) + x_margin]);
ylim(ax, [state_lb(2) - y_margin, state_ub(2) + y_margin]);

% Legend
legend(ax, 'Location', 'best', 'FontSize', 11, 'Box', 'on');
try
    leg = legend(ax);
    leg.BoxFaceAlpha = 0.95;
catch
end

hold(ax, 'off');

%% Save plot
output_path = fullfile(project_root, 'plots', 'simulation_results_2d.png');
exportgraphics(fig, output_path, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('Saved plot to: %s\n', output_path);

%% Helper function to parse config file
function config = parse_config(config_file)
    config = struct();
    
    fid = fopen(config_file, 'r');
    if fid == -1
        error('Could not open config file: %s', config_file);
    end
    
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            % Remove comments
            comment_idx = find(line == '#', 1);
            if ~isempty(comment_idx)
                line = line(1:comment_idx-1);
            end
            
            % Skip empty lines
            line = strtrim(line);
            if isempty(line)
                continue;
            end
            
            % Parse key = value
            eq_idx = find(line == '=', 1);
            if ~isempty(eq_idx)
                key = strtrim(line(1:eq_idx-1));
                value_str = strtrim(line(eq_idx+1:end));
                
                % Parse value (handle arrays and scalars)
                if contains(value_str, ',')
                    % Array
                    values = str2num(value_str);
                    config.(key) = values;
                else
                    % Scalar or string
                    num_val = str2double(value_str);
                    if ~isnan(num_val)
                        config.(key) = num_val;
                    else
                        config.(key) = value_str;
                    end
                end
            end
        end
    end
    
    fclose(fid);
end
