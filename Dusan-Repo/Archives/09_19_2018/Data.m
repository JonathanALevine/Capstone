classdef Data < handle
    properties
        file_name
        inputs = struct('structure', {}, 'parameter', {}, 'range', {})
        outputs = struct('port', {}, 'attribute', {})
        examples = struct('features', {}, 'labels', {})
        wavelengths = [1.3e-6, 1.7e-6]
    end
    methods
        function obj = Data
            obj.file_name = "H:\photonmind-master\Devices\" + input('Enter the FDTD file name: ', 's');

            while length(obj.inputs) < 1 || input('Add another input? Y/N: ', 's') == 'y'
                obj.inputs(end + 1).structure = input('Enter the structure name: ', 's');
                obj.inputs(end).parameter = input('Enter the parameter name: ', 's');
                user_range = input('Enter the parameter range as a 1x2 matrix: ');
                obj.inputs(end).range = [(user_range(1) - 0.1*diff(user_range)),...
                    (user_range(2) + 0.1*diff(user_range))];
            end

            while length(obj.outputs) < 1 || input('Add another output? Y/N: ', 's') == 'y'
                obj.outputs(end + 1).port = input('Enter the monitor/port name: ', 's');
                obj.outputs(end).attribute = input('Enter the monitor/port attribute: ', 's');
            end
        end
        
        function add_input(obj)
            obj.inputs(end + 1).structure = input('Enter the structure name: ', 's');
            obj.inputs(end).parameter = input('Enter the parameter name: ', 's');
            user_range = input('Enter the parameter range as a 1x2 matrix: ');
            obj.inputs(end).range = [(user_range(1) - 0.1*diff(user_range)),...
                (user_range(2) + 0.1*diff(user_range))];
        end
        
        function add_output(obj)
            obj.outputs(end + 1).port = input('Enter the monitor/port name: ', 's');
            obj.outputs(end).attribute = input('Enter the monitor/port attribute: ', 's');
        end
        
        function remove_input(obj, index)
            obj.inputs(index) = [];
        end
        
        function remove_output(obj, index)
            obj.outputs(index) = [];
        end
        
        function get_examples_random(obj, num_sim)
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');

            featureset = zeros(num_sim, length(obj.inputs));
            for m = 1:num_sim
                for n = 1:length(obj.inputs)
                    featureset(m, n) = obj.inputs(n).range(1) + diff(obj.inputs(n).range)*rand;
                end
            end
            
            v = waitbar(0, 'Acquiring data...');
            for m = 1:size(featureset, 1)
                waitbar(m/size(featureset, 1));
                labels = obj.simulate(featureset(m, :), h);
                obj.examples(end + 1).features = featureset(m, :);
                obj.examples(end).labels = labels;
            end
            close(v);
        end

        function get_examples_uniform(obj, resolution)
            featureset = zeros(length(obj.inputs), resolution^length(obj.inputs));
            if input(sprintf('This will run %d simulations. Proceed? Y/N: ', size(featureset, 2)), 's') ~= 'y'
                return;
            end
            
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');
            
            for m = 1:length(obj.inputs)
                sequence = linspace(obj.inputs(m).range(1), obj.inputs(m).range(2), resolution);
                sequence = repmat(sequence, [resolution^(length(obj.inputs) - m), 1]);
                featureset(m, :) = repmat(sequence(:)', [1, resolution^(m - 1)]);
            end
            featureset = featureset';
            
            v = waitbar(0, 'Acquiring data...');
            for m = 1:size(featureset, 1)
                code = strcat('load("',char(obj.file_name),'");',...
                    'switchtolayout;',...
                    'select("FDTD::ports");',...
                    'set("source mode", "mode 1");');
                appevalscript(h, code);
                labels = obj.simulate(featureset(m, :), h);
                obj.examples(end + 1).features = cat(2, 0, featureset(m, :));
                obj.examples(end).labels = labels;
                
                code = strcat('switchtolayout;',...
                    'select("FDTD::ports");',...
                    'set("source mode", "mode 2");');
                appevalscript(h, code);
                labels = obj.simulate(featureset(m, :), h);
                obj.examples(end + 1).features = cat(2, 1, featureset(m, :));
                obj.examples(end).labels = labels;
                
                waitbar(m/size(featureset, 1));
            end
            close(v);
        end
        
        function check_single(obj, features)
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');
            obj.simulate(features, h)
        end
        
        function remove_bad_spectrums(obj, T_min)
            m = 1;
            while m <= length(obj.examples)
                if abs(min(obj.examples(m).labels)) < abs(T_min)
                    obj.examples(m) = [];
                    m = m - 1;
                end
                m = m + 1;
            end
        end

        function labels = simulate(obj, features, h)
            % load file and get it ready for changes
%             code = strcat('load("',char(obj.file_name),'");',...
%                 'switchtolayout;');
%             appevalscript(h, code);

            % make (direct) changes from the features
            for n = 1:length(obj.inputs)
                code = strcat('select("',char(obj.inputs(n).structure),'");',...
                    'set("',char(obj.inputs(n).parameter),'", ',num2str(features(n)),');');
                appevalscript(h, code);
            end

            % run
            code = strcat('run;');
            appevalscript(h, code);

            % extract the labels
            labels = [];
            for n = 1:length(obj.outputs)
                code = strcat('port = getresult("FDTD::ports::',char(obj.outputs(n).port),'", "T");',...
                    'T = port.T;',...
                    'lam = port.lambda;');
                appevalscript(h, code);
                labels = cat(2, labels, fliplr(appgetvar(h, char(obj.outputs(n).attribute))'));
            end
        end
    end
end
