classdef DataParamSweep < Data
    properties
    end
    methods    
        function featureset = get_featureset(obj, resolution)
            featureset = zeros(length(obj.inputs), resolution^length(obj.inputs));
            for i = 1:length(obj.inputs)
                sequence = linspace(obj.inputs(i).range(1), obj.inputs(i).range(2), resolution);
                sequence = repmat(sequence, [resolution^(length(obj.inputs) - i), 1]);
                featureset(i, :) = repmat(sequence(:)', [1, resolution^(i - 1)]);
            end
        end
    end
end
