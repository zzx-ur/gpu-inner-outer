%% Visualize reachable_set.mat with Hyperbolic Constraints
%
% This script loads reachable_set.mat and visualizes the reachable set
% along with candidate set and hyperbolic constraint curves.
% Parameters are extracted from ../examples/hyperbolic_case.cfg
%
% Usage: Open and run this script in MATLAB (or press F5)

clear; close all; clc;

%% Configuration
% Determine project root (parent of plots/)
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

% Color scheme (optimized for contrast)
candidate_color  = [0.2, 0.2, 0.6]; % Deep purple-blue
candidate_alpha  = 0.8;
reachable_color  = [1.0, 0.2, 0.2]; % Light red
reachable_alpha  = 0.2;
azim             = 100;
elev             = 16;

%% Load reachable set data
mat_file = fullfile(project_root, 'plots', 'reachable_set.mat');
fprintf('Loading reachable set from %s...\n', mat_file);

if ~isfile(mat_file)
    error('Reachable set file not found: %s\nPlease run: python3 plots/postprocess_reachable_set.py', mat_file);
end

data = load(mat_file);

% Extract data from loaded structure
reachable_verts_phys = data.reachable_vertices;
reach_faces = data.reachable_faces + 1;  % Convert from 0-indexed to 1-indexed
candidate_verts_phys = data.candidate_vertices;
cand_faces_rect = data.candidate_faces + 1;  % Convert from 0-indexed to 1-indexed

% Get state space bounds for axis limits
lb = data.state_lb;
eta = data.state_eta;
shape = data.state_shape;

fprintf('✓ Loaded reachable set with %d vertices and %d faces\n', ...
    size(reachable_verts_phys, 1), size(reach_faces, 1));

%% Parse hyperbolic constraint parameters from config file
config_file = fullfile(project_root, 'examples', 'hyperbolic_case.cfg');
fprintf('Parsing constraint parameters from %s...\n', config_file);

% Default values
hyperbolic_a = 4.0;
hyperbolic_b = 4.0;
hyperbolic_c = 16.0;

% Read config file and extract parameters
if isfile(config_file)
    fid = fopen(config_file, 'r');
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            % Remove comments
            comment_idx = find(line == '#', 1);
            if ~isempty(comment_idx)
                line = line(1:comment_idx-1);
            end
            
            % Parse hyperbolic_a
            if contains(line, 'hyperbolic_a')
                parts = strsplit(line, '=');
                if length(parts) >= 2
                    hyperbolic_a = str2double(strtrim(parts{2}));
                end
            end
            
            % Parse hyperbolic_b
            if contains(line, 'hyperbolic_b')
                parts = strsplit(line, '=');
                if length(parts) >= 2
                    hyperbolic_b = str2double(strtrim(parts{2}));
                end
            end
            
            % Parse hyperbolic_c
            if contains(line, 'hyperbolic_c')
                parts = strsplit(line, '=');
                if length(parts) >= 2
                    hyperbolic_c = str2double(strtrim(parts{2}));
                end
            end
        end
    end
    fclose(fid);
    fprintf('✓ Extracted hyperbolic parameters: a=%.1f, b=%.1f, c=%.1f\n', ...
        hyperbolic_a, hyperbolic_b, hyperbolic_c);
else
    warning('Config file not found: %s. Using default hyperbolic parameters.', config_file);
end

%% Create plot
fig = figure('Color', 'white', 'Position', [100 100 1100 850]);
ax = axes('Parent', fig);
hold(ax, 'on');

% Plot candidate set
if size(candidate_verts_phys, 1) >= 8
    patch(ax, 'Faces', cand_faces_rect, 'Vertices', candidate_verts_phys, ...
          'FaceColor', candidate_color, 'FaceAlpha', candidate_alpha, ...
          'EdgeColor', 'k', 'LineWidth', 1.5);
end

% Plot reachable set
if ~isempty(reach_faces) && size(reachable_verts_phys, 1) > 0
    patch(ax, 'Faces', reach_faces, 'Vertices', reachable_verts_phys, ...
          'FaceColor', reachable_color, 'FaceAlpha', reachable_alpha, ...
          'EdgeColor', 'none');
end

%% Draw hyperbolic constraint curves at theta = -pi, 0, +pi
% Constraint 1: x1^2 - x2^2 <= a  =>  x1^2 = x2^2 + a
% Constraint 2: b*x2^2 - x1^2 <= c  =>  x2^2 = (x1^2 + c) / b

% Calculate intersection points
% At intersection: x1^2 - x2^2 = a AND b*x2^2 - x1^2 = c
% From constraint 1: x1^2 = x2^2 + a
% Substitute into constraint 2: b*x2^2 - (x2^2 + a) = c
%                               (b-1)*x2^2 = c + a
%                               x2^2 = (c + a) / (b - 1)
% Then: x1^2 = x2^2 + a = (c + a)/(b-1) + a = (c + a + a*(b-1))/(b-1) = (c + a*b)/(b-1)

x2_intersect_sq = (hyperbolic_c + hyperbolic_a) / (hyperbolic_b - 1);
x1_intersect_sq = (hyperbolic_c + hyperbolic_a * hyperbolic_b) / (hyperbolic_b - 1);

x1_intersect = sqrt(x1_intersect_sq);
x2_intersect = sqrt(x2_intersect_sq);

fprintf('Intersection points: x1=±%.3f, x2=±%.3f\n', x1_intersect, x2_intersect);

theta_vals = [-pi, 0, pi];

for theta = theta_vals
    
    % 1. Top arc (from constraint 2): from left intersection to right intersection
    x1_top = linspace(-x1_intersect, x1_intersect, 100);
    x2_top = sqrt((x1_top.^2 + hyperbolic_c) / hyperbolic_b);
    
    % 2. Right arc (from constraint 1): from top intersection to bottom intersection
    x2_right = linspace(x2_intersect, -x2_intersect, 100);
    x1_right = sqrt(x2_right.^2 + hyperbolic_a);
    
    % 3. Bottom arc (from constraint 2): from right intersection to left intersection
    x1_bottom = linspace(x1_intersect, -x1_intersect, 100);
    x2_bottom = -sqrt((x1_bottom.^2 + hyperbolic_c) / hyperbolic_b);
    
    % 4. Left arc (from constraint 1): from bottom intersection to top intersection
    x2_left = linspace(-x2_intersect, x2_intersect, 100);
    x1_left = -sqrt(x2_left.^2 + hyperbolic_a);
    
    % Concatenate to form complete closed loop
    x1_loop = [x1_top, x1_right, x1_bottom, x1_left];
    x2_loop = [x2_top, x2_right, x2_bottom, x2_left];
    
    % Plot the closed curve
    plot3(ax, x1_loop, x2_loop, theta * ones(size(x1_loop)), 'r-', 'LineWidth', 3);
end

%% Formatting
x_range = [lb(1), lb(1) + shape(1)*eta(1)];
y_range = [lb(2), lb(2) + shape(2)*eta(2)];
z_range = [lb(3), lb(3) + shape(3)*eta(3)];

axis(ax, [x_range, y_range, z_range]);
ax.Box = 'on';
view(ax, [azim, elev]);
grid(ax, 'on');

xlabel(ax, 'x', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(ax, 'y', 'FontSize', 14, 'FontWeight', 'bold');
zlabel(ax, '\theta', 'FontSize', 14, 'FontWeight', 'bold');

ax.FontSize = 12;
ax.FontWeight = 'bold';

title(ax, 'Candidate vs Reachable | Velocity: Projected', ...
      'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

% Legend
h_cand = patch(NaN, NaN, NaN, 'FaceColor', candidate_color, 'FaceAlpha', candidate_alpha);
h_reach = patch(NaN, NaN, NaN, 'FaceColor', reachable_color, 'FaceAlpha', reachable_alpha);
h_const = plot3(NaN, NaN, NaN, 'r-', 'LineWidth', 3);

leg = legend(ax, [h_cand, h_reach, h_const], ...
       {'Candidate', 'Reachable', 'Hyperbolic Constraint (\theta = -\pi, 0, +\pi)'}, ...
       'Location', 'northeast', 'FontSize', 12, 'Box', 'on');

try leg.BoxFaceAlpha = 0.95; catch; end
hold(ax, 'off');

%% Save plot
plots_dir = fullfile(project_root, 'plots');
if ~exist(plots_dir, 'dir'), mkdir(plots_dir); end

output_path = fullfile(plots_dir, 'hyperbolic_demo_reachable_set_from_mat.png');
exportgraphics(fig, output_path, 'Resolution', 300, 'BackgroundColor', 'white');
fprintf('✓ Saved plot to: %s\n', output_path);
