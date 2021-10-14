classdef Model < handle
    properties
        data
        examples = struct('train', {}, 'validate', {}, 'test', {})
        weights
        biases
        learning_rate = 0.001
        momentum = 0.5
        activation_function = 'sig'
        feature_ranges
        label_ranges
    end
    methods
        function obj = Model(data, num_hidden_neurons)
            obj.data = data;
            obj.examples(end + 1).train = data.examples(1:round(0.7*end));
            obj.examples(end).validate = data.examples(round(0.7*end) + 1:round(0.85*end));
            obj.examples(end).test = data.examples(round(0.85*end) + 1:end);
            obj.get_ranges;
            
            num_input_neurons = length(data.inputs);
            num_output_neurons = length(data.outputs);
            obj.weights{1} = normrnd(0, 1, [num_input_neurons, num_hidden_neurons])/num_input_neurons;
            obj.weights{2} = normrnd(0, 1, [num_hidden_neurons, num_output_neurons])/num_hidden_neurons;
            obj.biases{1} = normrnd(0, 1, [1, num_hidden_neurons])/num_input_neurons;
            obj.biases{2} = normrnd(0, 1, [1, num_output_neurons])/num_hidden_neurons;
        end
        
        function train(obj, iterations)
            v = waitbar(0, 'Training...');
            step = 0;
            error_list_train = zeros(1, iterations);
            
            for i = randi(length(obj.examples.train), 1, iterations)
                step = step + 1;
                waitbar(step/iterations)
                
                X = obj.scale(obj.examples.train(i).features, 'f')*obj.weights{1} + obj.biases{1};
                S = obj.ACT(X);
                Y = S*obj.weights{2} + obj.biases{2};
                Z = Y;
                
                error = mean(mean(abs(obj.scale(obj.examples.train(i).labels, 'l') - Y)));
                
                dEo = Z - obj.scale(obj.examples.train(i).labels, 'l');
                dZ = ones(size(Y));
                dY = S;
                dw2 = dY'*(dEo.*dZ);
                db2 = dEo.*dZ;
                
                dEh = dEo*obj.weights{2}';
                dS = obj.dACT(X);
                dX = obj.scale(obj.examples.train(i).features, 'f');
                dw1 = dX'*(dEh.*dS);
                db1 = dEh.*dS;
                
                obj.weights{1} = obj.weights{1} - obj.learning_rate*dw1;
                obj.weights{2} = obj.weights{2} - obj.learning_rate*dw2;
                obj.biases{1} = obj.biases{1} - obj.learning_rate*db1;
                obj.biases{2} = obj.biases{2} - obj.learning_rate*db2;
                
                error_list_train(step) = error;
            end
            plot(error_list_train);
            close(v);
        end
        
        function validation_error = validate(obj)
            validation_error = mean(mean(abs(obj.scale(reshape([obj.examples.validate.labels],...
                [length(obj.examples.validate(1).labels) length(obj.examples.validate)])', 'l')...
                - obj.infer(reshape([obj.examples.validate.features],...
                [length(obj.examples.validate(1).features) length(obj.examples.validate)])', false))));
        end
        
        function test_error = test(obj)
            test_error = mean(mean(abs(obj.scale(reshape([obj.examples.test.labels],...
                [length(obj.examples.test(1).labels) length(obj.examples.test)])', 'l')...
                - obj.infer(reshape([obj.examples.test.features],...
                [length(obj.examples.test(1).features) length(obj.examples.test)])', false))));
        end
        
        function Z = infer(obj, features, descale)
            switch nargin
                case 2
                    descale = true;
            end
            X = obj.scale(features, 'f')*obj.weights{1} + obj.biases{1};
            S = obj.ACT(X);
            Y = S*obj.weights{2} + obj.biases{2};
            Z = Y;
            if descale == true, Z = obj.descale(Z); end
        end
        
        function reset_weights(obj, num_hidden_neurons)
            rng('default');
            sow1 = size(obj.weights{1});
            sow2 = size(obj.weights{2});
            switch nargin
                case 1
                    num_hidden_neurons = sow1(2);
            end
            obj.weights{1} = normrnd(0, 1, [sow1(1), num_hidden_neurons])/sow1(1);
            obj.weights{2} = normrnd(0, 1, [num_hidden_neurons, sow2(2)])/num_hidden_neurons;
            obj.biases{1} = normrnd(0, 1, [1, num_hidden_neurons])/sow1(1);
            obj.biases{2} = normrnd(0, 1, [1, sow2(2)])/num_hidden_neurons;
        end
        
        function get_ranges(obj)
            features = reshape([obj.data.examples.features],...
                [length(obj.data.examples(1).features) length(obj.data.examples)])';
            labels = reshape([obj.data.examples.labels],...
                [length(obj.data.examples(1).labels) length(obj.data.examples)])';
            obj.feature_ranges = [min(features, [], 1); max(features, [], 1)];
            obj.label_ranges = [min(labels, [], 1); max(labels, [], 1)];
        end
        
        function y = scale(obj, values, type)
            switch type
                case 'f'
                    y = (values - obj.feature_ranges(1, :))./(obj.feature_ranges(2, :) - obj.feature_ranges(1, :));
                case 'l'
                    y = (values - obj.label_ranges(1, :))./(obj.label_ranges(2, :) - obj.label_ranges(1, :));
            end
        end
        
        function y = descale(obj, values)
            y = obj.label_ranges(1, :) + values.*(obj.label_ranges(2, :) - obj.label_ranges(1, :));
        end
        
        function y = ACT(obj, x)
            switch obj.activation_function
                case 'sig'
                    y = 1./(1 + exp(-x));
                case 'relu'
                    y = max(0, x);
                case 'tanh'
                    y = tanh(x);
            end
        end
        
        function y = dACT(obj, x)
            switch obj.activation_function
                case 'sig'
                    y = obj.ACT(x).*(1 - obj.ACT(x));
                case 'relu'
                    y = heaviside(x);
                case 'tanh'
                    y = 1 - obj.ACT(x).^2;
            end
        end
        
        function y = error(obj, calculated, target)
        end
        
        function y = derror(obj, calculated, target)
        end
    end
end
