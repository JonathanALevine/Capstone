
classdef Mind < handle
    properties
        data
        examples
        layers
        weights
        biases
        ratio
        sample_size
        batch_size
        optimizer
        feature_ranges
        label_ranges
        
        train_error
        validate_error
        test_error
    end
    methods
        function obj = Mind(data, shape, optimizer, ratio)
            obj.data = data;
            obj.ratio = ratio;
            obj.split_dataset;
            obj.get_data_ranges;
            obj.init_ANN(shape);
            obj.sample_size = length(obj.examples.train);
            obj.batch_size = obj.sample_size;
            obj.optimizer = Optimizer(optimizer, obj);
        end

        function split_dataset(obj)
            obj.examples = struct('train', {}, 'validate', {}, 'test', {});
            obj.examples(end + 1).train =...
                obj.data.examples(1:round(obj.ratio(1)*end));
            obj.examples(end).validate =...
                obj.data.examples(...
                (round(obj.ratio(1)*end) + 1):...
                round((obj.ratio(1) + obj.ratio(2))*end));
            obj.examples(end).test =...
                obj.data.examples(...
                round(((obj.ratio(1) + obj.ratio(2))*end) + 1):end);
        end
        
        function init_ANN(obj, shape)
            rng('default');
            
            obj.layers = Layer('input',...
                length(obj.data.examples(1).features), 'none');
            for n = 1:length(shape)
                obj.layers(end + 1) = Layer('hidden', shape(n));
                obj.weights{length(obj.layers) - 1} = normrnd(0, 1,...
                    [obj.layers(length(obj.layers) - 1).num_neurons,...
                    obj.layers(length(obj.layers)).num_neurons])...
                    /obj.layers(length(obj.layers) - 1).num_neurons;
                obj.biases{length(obj.layers) - 1} = normrnd(0, 1,...
                    [1, obj.layers(length(obj.layers)).num_neurons])...
                    /obj.layers(length(obj.layers) - 1).num_neurons;
            end
            obj.layers(end + 1) = Layer('output',...
                length(obj.data.examples(1).labels), 'none');
            obj.weights{length(obj.layers) - 1} = normrnd(0, 1,...
                [obj.layers(length(obj.layers) - 1).num_neurons,...
                obj.layers(length(obj.layers)).num_neurons])...
                /obj.layers(length(obj.layers) - 1).num_neurons;
            obj.biases{length(obj.layers) - 1} = normrnd(0, 1,...
                [1, obj.layers(length(obj.layers)).num_neurons])...
                /obj.layers(length(obj.layers) - 1).num_neurons;
        end
        
        function train(obj, num_epochs, test_data)
            switch nargin
                case 2
                    plot_test_error = false;
                case 3
                    plot_test_error = true;
            end
            
            format long;
            v = waitbar(0, 'Training...');
            error_list_train =...
                zeros(1, num_epochs*ceil(obj.sample_size/obj.batch_size));
            error_list_validate =...
                zeros(1, num_epochs*ceil(obj.sample_size/obj.batch_size));
            error_list_test =...
                zeros(1, num_epochs*ceil(obj.sample_size/obj.batch_size));
            
            features = reshape(...
                [obj.examples.train(1:obj.sample_size).features],...
                [length(obj.examples.train(1).features) obj.sample_size])';
            labels = reshape(...
                [obj.examples.train(1:obj.sample_size).labels],...
                [length(obj.examples.train(1).labels) obj.sample_size])';
            
            batches = struct('features', {}, 'labels', {});
            for n = 1:floor(obj.sample_size/obj.batch_size)
                batches(end + 1).features...
                    = features(((n - 1)*obj.batch_size + 1)...
                    :n*obj.batch_size, :);
                batches(end).labels...
                    = labels(((n - 1)*obj.batch_size + 1):n*obj.batch_size, :);
            end
            if n*obj.batch_size < obj.sample_size
                batches(end + 1).features...
                    = features((n*obj.batch_size + 1):obj.sample_size, :);
                batches(end).labels...
                    = labels((n*obj.batch_size + 1):obj.sample_size, :);
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
                        obj.layers(n).feed(obj.layers(n - 1),...
                            obj.weights{n - 1}, obj.biases{n - 1});
                    end
                    
                    derr = obj.layers(end).out - obj.scale(labels, 'l');
                    dout = ones(size(obj.layers(end).out));
                    dnet = obj.layers(end - 1).out;
                    obj.layers(end).dw = dnet'*(derr.*dout);
                    obj.layers(end).db...
                        = ones(size(features, 1), 1)'*(derr.*dout);
                    
                    for n = (length(obj.layers) - 1):-1:2
                        derr = derr*obj.weights{n}';
                        dout = obj.layers(n).dACT(obj.layers(n).net);
                        dnet = obj.layers(n - 1).out;
                        obj.layers(n).dw = dnet'*(derr.*dout);
                        obj.layers(n).db...
                            = ones(size(features, 1), 1)'*(derr.*dout);
                    end
                    
                    obj.optimizer.optimize(obj);
                    
                    error_list_train(m*length(batches) + k - 1)...
                        = obj.error;
                    error_list_validate(m*length(batches) + k - 1)...
                        = obj.validate;
                    if plot_test_error == true
                        err = obj.test(test_data);
                        error_list_test(m*length(batches) + k - 1)...
                            = err(1);
                    end
                end
            end
            
            obj.train_error = error_list_train;
            obj.validate_error = error_list_validate;
            obj.test_error = error_list_test;
            
            figure; hold on;
            box on;
            ylabel("Error");
            xlabel("Epochs");
            plot(error_list_train);
            plot(error_list_validate, 'r');
            legend("Training Error", "Validation Error");
            if plot_test_error == true
                plot(error_list_test, 'k');
                legend("Training Error", "Validation Error", "Test Error");
            end
            close(v);
        end
        
        function training_error = error(obj)
            labels = reshape([obj.examples.train.labels],...
                [length(obj.examples.train(1).labels)...
                length(obj.examples.train)])';
            predictions = obj.infer(reshape([obj.examples.train.features],...
                [length(obj.examples.train(1).features)...
                length(obj.examples.train)])');
            error = abs(labels - predictions);
            training_error = mean(mean(error));
        end
        
        function validation_error = validate(obj)
            labels = reshape([obj.examples.validate.labels],...
                [length(obj.examples.validate(1).labels)...
                length(obj.examples.validate)])';
            predictions = obj.infer(reshape([obj.examples.validate.features],...
                [length(obj.examples.validate(1).features)...
                length(obj.examples.validate)])');
            error = abs(labels - predictions);
            validation_error = mean(mean(error));
        end
        
        
        function test_error = test(obj, data)
            labels = reshape([data.examples.labels],...
                [length(data.examples(1).labels) length(data.examples)])';
            predictions = obj.infer(reshape([data.examples.features],...
                [length(data.examples(1).features)...
                length(data.examples)])');
            error = abs(labels - predictions)./labels;
            test_error = mean(mean(error));
        end

        function y = infer(obj, features, descale)
            switch nargin
                case 2
                    descale = true;
            end

            obj.layers(1).net = obj.scale(features, 'f');
            obj.layers(1).out = obj.layers(1).net;
            for n = 2:length(obj.layers)
                obj.layers(n).feed(obj.layers(n - 1),...
                    obj.weights{n - 1}, obj.biases{n - 1});
            end  
            y = obj.layers(end).out;
            if descale == true, y = obj.descale(y); end
        end

        function get_data_ranges(obj)
            features = reshape([obj.data.examples.features],...
                [length(obj.data.examples(1).features)...
                length(obj.data.examples)])';
            labels = reshape([obj.data.examples.labels],...
                [length(obj.data.examples(1).labels)...
                length(obj.data.examples)])';
            obj.feature_ranges = [min(features, [], 1); max(features, [], 1)];
            obj.label_ranges = [min(labels, [], 1); max(labels, [], 1)];
        end

        function y = scale(obj, values, type)
            switch type
                case 'f'
                    y = (values - obj.feature_ranges(1, :))...
                        ./(obj.feature_ranges(2, :) - obj.feature_ranges(1, :));
                case 'l'
                    y = (values - obj.label_ranges(1, :))...
                        ./(obj.label_ranges(2, :) - obj.label_ranges(1, :));
            end
        end

        function y = descale(obj, values)
            y = obj.label_ranges(1, :) +...
                values.*(obj.label_ranges(2, :) - obj.label_ranges(1, :));
        end
        
        % ----------------- %
        % TEMPORARY METHODS
        % ----------------- %
        
        function test_transmission_spectrum(obj, subset, example)
            figure; hold on;
            x = linspace(1.4e-6, 1.7e-6, length(subset(example).labels))*1e6;
            plot(x, -(obj.infer(subset(example).features)));
            plot(x, -(subset(example).labels));
            legend('Model', 'Simulation');
            xlabel("Wavelength (?m)");
            ylabel("Transmission (a.u.)")
            box on
        end
        
        function copy_mind(obj, source)
            obj.weights = source.weights;
            obj.biases = source.biases;
            obj.optimizer = source.optimizer;
        end
        
        function prune_examples(obj)
            m = 1;
            while m <= length(obj.examples.train)
                if obj.examples.train(m).labels < 0.15
                    obj.examples.train(m) = [];
                    m = m - 1;
                end
                m = m + 1;
            end
            
            m = 1;
            while m <= length(obj.examples.validate)
                if obj.examples.validate(m).labels < 0.15
                    obj.examples.validate(m) = [];
                    m = m - 1;
                end
                m = m + 1;
            end
            
            obj.sample_size = length(obj.examples.train);
            obj.batch_size = obj.sample_size;
        end
        
        function map_examples(obj, feature1, feature2)
            switch nargin
                case 1
                    feature1 = 1;
                    feature2 = 2;
            end
            
            features = reshape([obj.examples.train.features],...
                [length(obj.examples.train(1).features)...
                length(obj.examples.train)])';
            scatter(features(:, feature1), features(:, feature2), 'filled');
        end
        
        function count = count_good_examples(obj, threshold)
            count = 0;
            for n = 1:length(obj.examples.train)
                if abs(max(obj.examples.train(n).labels)) >= threshold
                    count = count + 1;
                end
            end
        end
    end
end
