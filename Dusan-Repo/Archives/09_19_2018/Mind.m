classdef Mind < handle
    properties
        data
        examples
        layers
        weights
        biases
        wdeltas
        bdeltas
        sample_size
        batch_size
        learning_rate = 0.0001
        momentum = 0.9
        feature_ranges
        label_ranges
    end
    methods
        function obj = Mind(data)
            obj.data = data;
            obj.data.examples = obj.data.examples(randperm(length(obj.data.examples)));
            obj.split_dataset;
            obj.get_data_ranges;
            obj.init_ANN;
            obj.sample_size = length(obj.examples.train);
            obj.batch_size = obj.sample_size;
        end

        function split_dataset(obj, ratio)
            switch nargin
                case 1
                    ratio = [0.85, 0.15, 0];
            end
            obj.examples = struct('train', {}, 'validate', {}, 'test', {});
            obj.examples(end + 1).train = obj.data.examples(1:round(ratio(1)*end));
            obj.examples(end).validate = obj.data.examples((round(ratio(1)*end) + 1):round((ratio(1) + ratio(2))*end));
            obj.examples(end).test = obj.data.examples(round(((ratio(1) + ratio(2))*end) + 1):end);
        end
        
        function init_ANN(obj)
            rng('default');
            
            obj.layers = Layer('input', length(obj.data.examples(1).features), 'none');
            
            while length(obj.layers) < 2 || input('Add another hidden layer? Y/N: ', 's') == 'y'
                obj.layers(end + 1) = Layer('hidden');
                obj.weights{length(obj.layers) - 1} = normrnd(0, 1,...
                    [obj.layers(length(obj.layers) - 1).num_neurons, obj.layers(length(obj.layers)).num_neurons])...
                    /obj.layers(length(obj.layers) - 1).num_neurons;
                obj.biases{length(obj.layers) - 1} = normrnd(0, 1,...
                    [1, obj.layers(length(obj.layers)).num_neurons])...
                    /obj.layers(length(obj.layers) - 1).num_neurons;
                obj.wdeltas{length(obj.layers) - 1} = zeros(obj.layers(length(obj.layers) - 1).num_neurons, obj.layers(length(obj.layers)).num_neurons);
                obj.bdeltas{length(obj.layers) - 1} = zeros(1, obj.layers(length(obj.layers)).num_neurons);
            end
            
            obj.layers(end + 1) = Layer('output', length(obj.data.examples(1).labels), 'none');
            obj.weights{length(obj.layers) - 1} = normrnd(0, 1,...
                [obj.layers(length(obj.layers) - 1).num_neurons, obj.layers(length(obj.layers)).num_neurons])...
                /obj.layers(length(obj.layers) - 1).num_neurons;
            obj.biases{length(obj.layers) - 1} = normrnd(0, 1,...
                [1, obj.layers(length(obj.layers)).num_neurons])...
                /obj.layers(length(obj.layers) - 1).num_neurons;
            obj.wdeltas{length(obj.layers) - 1} = zeros(obj.layers(length(obj.layers) - 1).num_neurons, obj.layers(length(obj.layers)).num_neurons);
            obj.bdeltas{length(obj.layers) - 1} = zeros(1, obj.layers(length(obj.layers)).num_neurons);
        end
        
        function train(obj, num_epochs)
            format long;
            v = waitbar(0, 'Training...');
            error_list_train = zeros(1, num_epochs*ceil(obj.sample_size/obj.batch_size));
            error_list_validate = zeros(1, num_epochs*ceil(obj.sample_size/obj.batch_size));
            
            features = reshape([obj.examples.train(1:obj.sample_size).features],...
                [length(obj.examples.train(1).features) obj.sample_size])';
            labels = reshape([obj.examples.train(1:obj.sample_size).labels],...
                [length(obj.examples.train(1).labels) obj.sample_size])';
            
            batches = struct('features', {}, 'labels', {});
            for n = 1:floor(obj.sample_size/obj.batch_size)
                batches(end + 1).features = features(((n - 1)*obj.batch_size + 1):n*obj.batch_size, :); %#ok
                batches(end).labels = labels(((n - 1)*obj.batch_size + 1):n*obj.batch_size, :);
            end
            if n*obj.batch_size < obj.sample_size
                batches(end + 1).features = features((n*obj.batch_size + 1):obj.sample_size, :);
                batches(end).labels = labels((n*obj.batch_size + 1):obj.sample_size, :);
            end
            
            for m = 1:num_epochs
                waitbar(m/num_epochs);
                for k = 1:length(batches)
                    ix = randperm(size(batches(k).features, 1));
                    features = batches(k).features;
                    features = features(ix, :);
                    labels = batches(k).labels;
                    labels = labels(ix, :);
                    
                    obj.layers(1).net = obj.scale(features, 'f');
                    obj.layers(1).out = obj.layers(1).net;
                    for n = 2:length(obj.layers)
                        obj.layers(n).feed(obj.layers(n - 1), obj.weights{n - 1}, obj.biases{n - 1});
                    end
                    
                    error = mean(mean(abs(obj.scale(labels, 'l') - obj.layers(end).out)));
                    
                    derr = obj.layers(end).out - obj.scale(labels, 'l');
                    dout = ones(size(obj.layers(end).out));
                    dnet = obj.layers(end - 1).out;
                    obj.layers(end).dw = dnet'*(derr.*dout);
                    obj.layers(end).db = ones(size(features, 1), 1)'*(derr.*dout);
                    
                    for n = (length(obj.layers) - 1):-1:2
                        derr = derr*obj.weights{n}';
                        dout = obj.layers(n).dACT(obj.layers(n).net);
                        dnet = obj.layers(n - 1).out;
                        obj.layers(n).dw = dnet'*(derr.*dout);
                        obj.layers(n).db = ones(size(features, 1), 1)'*(derr.*dout);
                    end
                    
                    for n = 1:length(obj.weights)
                        obj.weights{n} = obj.weights{n} - obj.learning_rate*obj.layers(n + 1).dw - obj.momentum*obj.wdeltas{n};
                        obj.biases{n} = obj.biases{n} - obj.learning_rate*obj.layers(n + 1).db - obj.momentum*obj.bdeltas{n};
                        obj.wdeltas{n} = obj.learning_rate*obj.layers(n + 1).dw + obj.momentum*obj.wdeltas{n};
                        obj.bdeltas{n} = obj.learning_rate*obj.layers(n + 1).db + obj.momentum*obj.bdeltas{n};
                    end
                    
                    error_list_train(m*length(batches) + k - 1) = error;
                    error_list_validate(m*length(batches) + k - 1) = obj.validate;
                end
            end
            figure; hold on;
            plot(error_list_train);
            box on;
            ylabel("Error");
            xlabel("Epochs");
            plot(error_list_validate, 'r');
            legend("Training Error","Validation Error");
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

        function y = infer(obj, features, descale)
            switch nargin
                case 2
                    descale = true;
            end

            obj.layers(1).net = obj.scale(features, 'f');
            obj.layers(1).out = obj.layers(1).net;
            for n = 2:length(obj.layers)
                obj.layers(n).feed(obj.layers(n - 1), obj.weights{n - 1}, obj.biases{n - 1});
            end
            y = obj.layers(end).out;
            if descale == true, y = obj.descale(y); end
        end

        function reset_weights(obj)
            rng('default');
            for n = 1:length(obj.weights)
                obj.weights{n} = normrnd(0, 1, [obj.layers(n).num_neurons, obj.layers(n + 1).num_neurons])...
                /obj.layers(n).num_neurons;
                obj.biases{n} = normrnd(0, 1, [1, obj.layers(n + 1).num_neurons])...
                /obj.layers(n).num_neurons;
                obj.wdeltas{n} = zeros(obj.layers(n).num_neurons, obj.layers(n + 1).num_neurons);
                obj.bdeltas{n} = zeros(1, obj.layers(n + 1).num_neurons);
            end
        end

        function get_data_ranges(obj)
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
        
        function test_inference(obj, subset, example)
            subset(example).features
            obj.infer(subset(example).features)
            subset(example).labels
        end

        function test_transmission_spectrum(obj, subset, example)
            figure; hold on;
            x = linspace(obj.data.wavelengths(1), obj.data.wavelengths(2), length(subset(example).labels))*1e6;
            plot(x, -(obj.infer(subset(example).features)));
            plot(x, -(subset(example).labels));
            legend('Model', 'Simulation');
            xlabel("Wavelength (µm)");
            ylabel("Transmission (a.u.)")
            box on
        end

        function publish(obj)
            activation_functions = cell(1, length(obj.layers));
            for n = 1:length(obj.layers)
                activation_functions{n} = obj.layers(n).activation_function;
            end
            new_model = Model(obj.data.inputs, obj.data.outputs, obj.weights, obj.biases,...
                activation_functions, obj.feature_ranges, obj.label_ranges); %#ok
            save('pm-pic.mat', 'new_model');
        end
    end
end
