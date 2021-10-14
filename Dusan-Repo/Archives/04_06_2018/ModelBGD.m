classdef ModelBGD < Model
    properties
    end
    methods
        function obj = ModelBGD(data, num_hidden_neurons)
            obj@Model(data, num_hidden_neurons);
        end
        
        function train(obj, num_epochs)
            v = waitbar(0, 'Training...');
            error_list_train = zeros(1, num_epochs);
            error_list_validate = zeros(1, num_epochs);
            
            features = reshape([obj.examples.train.features],...
                [length(obj.examples.train(1).features) length(obj.examples.train)])';
            labels = reshape([obj.examples.train.labels],...
                [length(obj.examples.train(1).labels) length(obj.examples.train)])';
            
            for i = 1:num_epochs
                waitbar(i/num_epochs)
                
                X = obj.scale_features(features)*obj.weights{1} + obj.biases{1};
                S = obj.ACT(X);
                Y = S*obj.weights{2} + obj.biases{2};
                Z = Y;
                
                error = mean(mean(abs(obj.scale_labels(labels) - Y)));
                
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
