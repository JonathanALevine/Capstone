classdef Model < handle
    properties
        data
        examples = struct('train', {}, 'validate', {}, 'test', {});
        weights
        activation_function = 'sig'
    end
    methods
        function obj = Model(data, num_hidden_neurons)
            obj.data = data;
            obj.examples(end + 1).train = data.examples(1:round(0.7*end));
            obj.examples(end).validate = data.examples(round(0.7*end) + 1:round(0.85*end));
            obj.examples(end).test = data.examples(round(0.85*end) + 1:end);
            
            num_input_neurons  = length(data.inputs);
            switch nargin
                case 1
                    num_hidden_neurons = input('Enter the number of hidden neurons: ');
            end
            num_output_neurons = length(data.outputs);
            obj.weights{1} = rand(num_input_neurons, num_hidden_neurons)/num_hidden_neurons;
            obj.weights{2} = rand(num_hidden_neurons, num_output_neurons)/num_hidden_neurons;
        end
        
        function train(obj, epochs)
            % formatting for small numbers
            format long;
            
            % setup the progress bar
            v = waitbar(0, 'Training...');
            waitbar_step = 0;
            
            % hyperparameters
            alpha = 0.1;
            
            for i = randi(length(obj.examples.train), 1, epochs)
%             for i = randi(20, 1, epochs)
                waitbar_step = waitbar_step + 1;
                waitbar(waitbar_step/epochs)
                
                % forward propagation
                X = obj.scale_features(obj.examples.train(i).features)*obj.weights{1};
                S = obj.ACT(X);
                Y = S*obj.weights{2};
                
                error = obj.scale_labels(obj.examples.train(i).labels) - Y;
                
                % backward propagation
                DY  = obj.dACT(Y).*error;
                dw2 = alpha*DY.*S';
                DX  = obj.weights{2}*DY'.*obj.dACT(X)';
                dw1 = alpha*obj.scale_features(obj.examples.train(i).features)'*DX';
                
                obj.weights{1} = obj.weights{1} + dw1;
                obj.weights{2} = obj.weights{2} + dw2;
            end
            close(v);
        end
        
        function MSE = validate(obj)
            L2_sum = 0;
            for i = 1:length(obj.examples.validate)
                L2 = (obj.examples.validate(i).labels - obj.infer(obj.examples.validate(i).features)).^2;
                
                L2_sum = L2_sum + L2;
            end
            MSE = L2_sum/i;
        end
        
        function MSE = test_full(obj)
            L2_sum = 0;
            for i = 1:length(obj.examples.test)
                L2 = (obj.examples.test(i).labels - obj.infer(obj.examples.test(i).features)).^2;
                
                L2_sum = L2_sum + L2;
            end
            MSE = L2_sum/i;
        end
        
        function L2 = test_single(obj, features)
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');
            
            labels = obj.data.simulate(features, h);
            
            L2 = (labels - obj.infer(features)).^2;
        end
        
        function Y = infer(obj, features)
            X = obj.scale_features(features)*obj.weights{1};
            S = obj.ACT(X);
            Y = S*obj.weights{2};
            Y = obj.descale_labels(Y);
        end
        
        function reset_weights(obj, nhid)
            sw1 = size(obj.weights{1});
            sw2 = size(obj.weights{2});
            obj.weights{1} = rand(sw1(1), nhid)/nhid;
            obj.weights{2} = rand(nhid, sw2(2))/nhid;
        end
        
        function ranges = get_label_scaling_ranges(obj)
            arr = [obj.examples.train.labels];
            ranges = zeros(length(obj.examples.train(1).labels), 2);
            for i = 1:length(obj.examples.train(1).labels)
                ranges(i, 1) = min(arr(i:length(obj.examples.train(1).labels):end));
                ranges(i, 2) = max(arr(i:length(obj.examples.train(1).labels):end));
            end
        end
        
        function y = scale_features(obj, features)
            y = zeros(1, length(features));
            for i = 1:length(features)
                y(i) = (features(i) - obj.data.inputs(i).range(1))...
                    /(obj.data.inputs(i).range(2) - obj.data.inputs(i).range(1));
            end
        end
        
        function y = scale_labels(obj, labels)
            ranges = obj.get_label_scaling_ranges;
            y = zeros(1, length(labels));
            for i = 1:length(labels)
                y(i) = (labels(i) - ranges(i, 1))/(ranges(i, 2) - ranges(i, 1));
            end
        end
        
        function y = descale_labels(obj, labels)
            ranges = obj.get_label_scaling_ranges;
            y = zeros(1, length(labels));
            for i = 1:length(labels)
                y(i) = ranges(i, 1) + labels(i)*(ranges(i, 2) - ranges(i, 1));
            end
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
    end
end
