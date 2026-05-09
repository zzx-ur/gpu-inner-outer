classdef H5Utils
    % H5Utils - Static utility methods for HDF5 file operations
    
    methods (Static)
        function exists = is_dataset(filepath, dataset_path)
            % is_dataset - Check if a dataset exists at the given path
            %
            % Inputs:
            %   filepath      - Path to HDF5 file
            %   dataset_path  - Full path to dataset (e.g., '/state/shape')
            %
            % Output:
            %   exists - true if dataset exists, false otherwise
            
            try
                info = h5info(filepath, dataset_path);
                exists = strcmp(info.Type, 'dataset');
            catch
                exists = false;
            end
        end
        
        function datasets = list_datasets(filepath, group_path)
            % list_datasets - List all datasets in an HDF5 group
            %
            % Inputs:
            %   filepath     - Path to HDF5 file
            %   group_path   - Path to group (e.g., '/result/gpu')
            %
            % Output:
            %   datasets - Cell array of dataset names
            
            try
                info = h5info(filepath, group_path);
                datasets = cellfun(@(x) x.Name, {info.Datasets.Name}, 'UniformOutput', false);
            catch
                datasets = {};
            end
        end
        
        function groups = list_groups(filepath, parent_path)
            % list_groups - List all subgroups in an HDF5 group
            %
            % Inputs:
            %   filepath     - Path to HDF5 file
            %   parent_path  - Path to parent group
            %
            % Output:
            %   groups - Cell array of group names
            
            try
                info = h5info(filepath, parent_path);
                groups = cellfun(@(x) x.Name, {info.Groups.Name}, 'UniformOutput', false);
            catch
                groups = {};
            end
        end
    end
end
