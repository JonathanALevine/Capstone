classdef Device < handle
    properties
        model
        parameters
        attributes
    end
    methods
        % object constructor
        function obj = Device(model)
            obj.model = model;
        end
        
        % sweep all features until a certain label is found
        function param_sweep(obj, res)
            % create the feature matrix
            sweeps = zeros(length(obj.model.inputs), res^length(obj.model.inputs));
            for i = 1:length(obj.model.inputs)
                arr = linspace(obj.model.inputs(i).range(1), obj.model.inputs(i).range(2), res);
                stretch = repmat(arr, [res^(length(obj.model.inputs) - i), 1]);
                sweeps(i, :) = repmat(stretch(:)', [1, res^(i - 1)]);
            end
            
            % check for request
            for i = 1:length(sweeps)
                labels = obj.model.infer(sweeps(:, i));
                condition = labels(2) > 1.54e-6 & labels(2) < 1.56e-6;
                if condition
                    sweeps(:, 1) %#ok
                    labels %#ok
                    if input('Would you like to set this device? Y/N [Y]: ', 's') == 'y'
                        obj.parameters = sweeps(:, 1);
                        obj.attributes = labels;
                    end
                end
            end
        end
    end
end
