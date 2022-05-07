classdef Model_Committee < Model
    properties
        examples = {};
    end
    methods
        function obj = Model(data, num_hidden_neurons)
            obj.data = data;
            obj.examples(end + 1) = data.examples(1:round(0.7*end));
            
            num_input_neurons  = length(data.inputs);
            switch nargin
                case 1
                    num_hidden_neurons = input('Enter the number of hidden neurons: ');
            end
            num_output_neurons = length(data.outputs);
            obj.weights{1} = rand(num_input_neurons, num_hidden_neurons)/num_hidden_neurons;
            obj.weights{2} = rand(num_hidden_neurons, num_output_neurons)/num_hidden_neurons;
        end  
    end
end
