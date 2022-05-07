classdef Model < handle
    properties
        file
        inputs = struct('structure', {}, 'parameter', {}, 'range', {}');
        outputs = struct('port', {}, 'attribute', {});
        data_train = struct('features', {}, 'labels', {});
        data_validate = struct('features', {}, 'labels', {});
        data_test = struct('features', {}, 'labels', {});
        weights
        act = 'sig'
    end
    methods
        function obj = Model
            obj.file = input('Enter the file path: ', 's');
            
            str = 'y';
            while str == 'y'
                obj.inputs(end + 1).structure = input('Enter the structure name: ', 's');
                obj.inputs(end).parameter = input('Enter the parameter name: ', 's');
                range = input('Enter the range as a 1x2 matrix: ');
                obj.inputs(end).range = [(range(1) - 0.1*diff(range)), (range(2) + 0.1*diff(range))];
                str = input('Do you want more inputs? Y/N [Y]: ', 's');
            end
            
            str = 'y';
            while str == 'y'
                obj.outputs(end + 1).port = input('Enter the port number: ', 's');
                obj.outputs(end).attribute = input('Enter the attribute: ', 's');
                str = input('Do you want more outputs? Y/N [Y]: ', 's');
            end
            
            ninp = length(obj.inputs);
            nhid = input('Enter the number of hidden neurons: ');
            nout = length(obj.outputs);
            obj.weights{1} = rand(ninp, nhid)/nhid;
            obj.weights{2} = rand(nhid, nout)/nhid;
        end
        
        % automated data acquisition from Lumerical FDTD
        function get_data(obj, num_sim)
            % add Lumerical MATLAB API path and open FDTD session
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');
            
            % simulate with FDTD
            for i = 1:num_sim
                % randomize parameters within ranges
                features = zeros(1, length(obj.inputs));
                for j = 1:length(obj.inputs)
                    features(j) = obj.inputs(j).range(1) + diff(obj.inputs(j).range)*rand;
                end
                
                % run simulation
                labels = obj.FDTD(features, h);
                
                % add values to the data structure
                obj.data_train(end + 1) = struct('features', {features}, 'labels', {labels});
            end
        end
        
        % run a single FDTD simulation
        function labels = FDTD(obj, features, h)
            % navigate to file and switch to layout view
            code = strcat('load("',char(obj.file),'");',...
                'switchtolayout;');
            appevalscript(h, code);
            
            % change the features
            for i = 1:length(obj.inputs)
                code = strcat('select("',char(obj.inputs(i).structure),'");',...
                    'set("',char(obj.inputs(i).parameter),'", ',num2str(features(i)),');');
                appevalscript(h, code);
            end
            
            % run the simulation
            code = strcat('run;');
            appevalscript(h, code);
            
            % get the requested outputs
            labels = zeros(1, length(obj.outputs));
            for i = 1:length(obj.outputs)
                code = strcat('port = getresult("FDTD::ports::port',char(obj.outputs(i).port),'","T");',...
                    'T = port.T;',...
                    'T_min = min(port.T);',...
                    'lam_T_min = port.lambda(find(port.T, min(port.T)));',...
                    'T_max = max(port.T);',...
                    'lam_T_max = port.lambda(find(port.T, max(port.T)));');
                appevalscript(h, code);
                data = appgetvar(h, char(obj.outputs(i).attribute)); % clunky
                labels(i:length(data)) = data; % be careful: arrays should be last
            end
        end
        
        % train the model
        function train(obj, num_examples)
            % formatting
            format long;
            
            % setup the progress bar
            v = waitbar(0, 'Training...');
            step = 0;
            
            % hyperparameters
            alpha = 1;
            
            % training loop
            iters = 1;
            for i = randi(2, 1, num_examples)
                % update progress bar
                step = step + 1;
                waitbar(step/num_examples)
                for j = 1:iters
                    % forward propagation
                    X = obj.iscale(obj.data_train(i).features)*obj.weights{1};
                    S = obj.ACT(X);
                    Y = S*obj.weights{2};
                    obj.oscale(obj.data_train(i).labels);
                    
                    % check the error
                    error = 0.5*(obj.oscale(obj.data_train(i).labels) - Y).^2;

                    % backward propagation
                    DY  = obj.dACT(Y).*error;
                    dw2 = alpha*DY.*S';
                    DX  = obj.weights{2}*DY'.*obj.dACT(X)';
                    dw1 = alpha*obj.iscale(obj.data_train(i).features)'*DX';
                    
                    % update weights
                    obj.weights{1} = obj.weights{1} + dw1;
                    obj.weights{2} = obj.weights{2} + dw2;
                end
            end
            
            % close progress bar
            close(v);
        end
        
        % this tests the model against a set of unseen data
        function MSE = test_full(obj)
            L2_sum = 0;
            for i = 1:length(obj.data_test)
                % calculate L2 loss
                L2 = (obj.data_test(i).labels - obj.infer(obj.data_test(i).features)).^2;
                
                % update sum of L2 loss
                L2_sum = L2_sum + L2;
            end
            
            % calculate mean square error
            MSE = L2_sum/i;
        end
        
        % this tests a single unseen device and checks with FDTD
        function L2 = test_single(obj, features)
            % add Lumerical MATLAB API path and open FDTD session
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');
            
            % perform simulation
            labels = obj.FDTD(features, h);
            
            % calculate L2 loss
            L2 = (labels - obj.infer(features)).^2;
        end
        
        % inference with scaling
        function Y = infer(obj, features)
            X = obj.iscale(features)*obj.weights{1};
            S = obj.ACT(X);
            Y = S*obj.weights{2};
            Y = obj.de_oscale(Y);
        end
        
        % create validation and test data
        function partition_data(obj)
            obj.data_test = obj.data_train((end - round(0.2*end)):end);
            obj.data_train((end - round(0.2*end)):end) = [];
            obj.data_validate = obj.data_train((end - round(0.2*end)):end);
            obj.data_train((end - round(0.2*end)):end) = [];
        end
        
        % reset weights
        function reset_weights(obj, nhid)
            sw1 = size(obj.weights{1});
            sw2 = size(obj.weights{2});
            obj.weights{1} = rand(sw1(1), nhid)/10;
            obj.weights{2} = rand(nhid, 100)/10;
        end
        
        % gets ranges for proper scaling
        function ranges = get_scaling_ranges(obj)
            arr = [obj.data_train.labels];
            ranges = zeros(length(obj.data_train(1).labels), 2);
            for i = 1:length(obj.data_train(1).labels)
                ranges(i, 1) = min(arr(i:length(obj.data_train(1).labels):end));
                ranges(i, 2) = max(arr(i:length(obj.data_train(1).labels):end));
            end
        end
        
        % scale features
        function y = iscale(obj, features)
            y = zeros(1, length(features));
            for i = 1:length(features)
                y(i) = (features(i) - obj.inputs(i).range(1))/(obj.inputs(i).range(2) - obj.inputs(i).range(1));
            end
        end
        
        % scale labels
        function y = oscale(obj, labels)
            ranges = obj.get_scaling_ranges;
            y = zeros(1, length(labels));
            for i = 1:length(labels)
                y(i) = (labels(i) - ranges(i, 1))/(ranges(i, 2) - ranges(i, 1));
            end
        end
        
        % descale labels
        function y = de_oscale(obj, labels)
            ranges = obj.get_scaling_ranges;
            y = zeros(1, length(labels));
            for i = 1:length(labels)
                y(i) = ranges(i, 1) + labels(i)*(ranges(i, 2) - ranges(i, 1));
            end
        end
        
        % activation function
        function y = ACT(obj, x)
            switch obj.act
                case 'sig'
                    y = 1./(1 + exp(-x));
                case 'relu'
                    y = max(0, x);
                case 'tanh'
                    y = tanh(x);
            end
        end
        
        % derivative of activation function
        function y = dACT(obj, x)
            switch obj.act
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
