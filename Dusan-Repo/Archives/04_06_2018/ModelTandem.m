classdef ModelTandem < Model
    properties
        model
    end
    methods
        function obj = ModelTandem(model, num_hidden_neurons)
            obj@Model(model.data, num_hidden_neurons);
            obj.model = model;
            
            for n = 1:length(obj.examples.train)
                tmp = obj.examples.train(n).features;
                obj.examples.train(n).features = obj.examples.train(n).labels;
                obj.examples.train(n).labels = tmp;
            end
            
            for n = 1:length(obj.examples.validate)
                tmp = obj.examples.validate(n).features;
                obj.examples.validate(n).features = obj.examples.validate(n).labels;
                obj.examples.validate(n).labels = tmp;
            end
            
            for n = 1:length(obj.examples.test)
                tmp = obj.examples.test(n).features;
                obj.examples.test(n).features = obj.examples.test(n).labels;
                obj.examples.test(n).labels = tmp;
            end
            
            obj.weights{1} = obj.model.weights{2}';
            obj.weights{2} = obj.model.weights{1}';
            obj.biases{1} = obj.model.biases{1};
            obj.biases{2} = normrnd(0, 1, [1, size(obj.weights{2}, 2)])/size(obj.weights{2}, 1);
            
            obj.biases{2} = normrnd(0, 1, [1, 3]);
            obj.reset_weights;
        end
        
        function train(obj, epochs)
            v = waitbar(0, 'Training...');
            error_list_train = zeros(1, epochs);
            error_list_validate = zeros(1, epochs);
            
            features = reshape([obj.examples.train.features],...
                [length(obj.examples.train(1).features) length(obj.examples.train)])';
            labels = reshape([obj.examples.train.labels],...
                [length(obj.examples.train(1).labels) length(obj.examples.train)])';
            
            for i = 1:epochs
                waitbar(i/epochs)
                
                X = obj.scale_features(features)*obj.weights{1} + obj.biases{1};
                S = obj.ACT(X);
                Y = S*obj.weights{2} + obj.biases{2};
                Z = Y;
                
                obj.model.infer(Z, false);
                error = mean(mean(abs(obj.model.infer(Z, false) - features)));
                
                dEo = Z - obj.scale_labels(labels);
                dZ = ones(size(Y));
                dY = S;
                dw2 = dY'*(dEo.*dZ);
                db2 = ones(length(features), 1)'*(dEo.*dZ);
                
                dEh = dEo*obj.weights{2}';
                dS = obj.dACT(X);
                dX = obj.scale_features(features);
                dw1 = dX'*(dEh.*dS);
                db1 = ones(length(features), 1)'*(dEh.*dS);
                
                obj.weights{1} = obj.weights{1} - obj.alpha*dw1;
                obj.weights{2} = obj.weights{2} - obj.alpha*dw2;
                obj.biases{1} = obj.biases{1} - obj.alpha*db1;
                obj.biases{2} = obj.biases{2} - obj.alpha*db2;
                
                error_list_train(i) = error;
                error_list_validate(i) = obj.validate;
            end
            figure; hold on;
            plot(error_list_train);
            plot(error_list_validate, 'r');
            legend('Training Error', 'Validation Error');
            close(v);
        end
    end
end
