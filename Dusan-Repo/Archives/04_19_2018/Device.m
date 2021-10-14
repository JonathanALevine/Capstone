classdef Device < handle
    properties
        model
        features
    end
    methods
        function obj = Device(model)
            obj.model = model;
        end
        
        function parameter_sweep(obj, resolution)
            % create the sweep matrix
            sweep = zeros(length(obj.model.inputs), resolution^length(obj.model.inputs));
            for i = 1:length(obj.model.inputs)
                sequence = linspace(obj.model.inputs(i).range(1), obj.model.inputs(i).range(2), resolution);
                sequence = repmat(sequence, [resolution^(length(obj.model.inputs) - i), 1]);
                sweep(i, :) = repmat(sequence(:)', [1, resolution^(i - 1)]);
            end
            
            for i = 1:length(sweep)
                labels = obj.model.infer(sweep(:, i));
                condition = labels(2) > 1.54e-6 & labels(2) < 1.56e-6;
                if condition
                    sweep(:, 1) %#ok
                    labels %#ok
                    if input('Would you like to set this device? y/n: ', 's') == 'y'
                        obj.features = sweep(:, 1);
                    end
                end
            end
        end
    end
end
