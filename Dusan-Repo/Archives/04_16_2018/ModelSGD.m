classdef ModelSGD < Model
    properties
    end
    methods
        function obj = ModelSGD(data, num_hidden_neurons)
            obj@Model(data, num_hidden_neurons);
        end
    end
end
