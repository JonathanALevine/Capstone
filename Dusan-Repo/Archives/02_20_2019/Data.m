
classdef Data < handle
    properties
        file_name
        inputs = struct('structure', {}, 'parameter', {}, 'range', {})
        outputs = struct('port', {}, 'attribute', {})
        examples = struct('features', {}, 'labels', {})
    end
    methods
        function obj = Data
            obj.file_name = "H:\photonmind-master\Devices\" + input('Enter the file name: ', 's');

            while length(obj.inputs) < 1 || input('Add another input? Y/N: ', 's') == 'y'
                obj.inputs(end + 1).structure = input('Enter the structure name: ', 's');
                obj.inputs(end).parameter = input('Enter the parameter name: ', 's');
                user_range = input('Enter the parameter range as a 1x2 matrix: ');
                obj.inputs(end).range = [(user_range(1) - 0.1*diff(user_range)),...
                    (user_range(2) + 0.1*diff(user_range))];
            end
        end
        
        function shuffle(obj)
            obj.examples = obj.examples(randperm(length(obj.examples)));
        end
        
        function get_examples_random(obj, num_sim)
            path(path, 'C:\Program Files\Lumerical\fdtd\api\matlab');
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
            
            path(path, 'C:\Program Files\Lumerical\device\api\matlab');
            h = appopen('fdtd');
            
            for m = 1:length(obj.inputs)
                sequence = linspace(obj.inputs(m).range(1), obj.inputs(m).range(2), resolution);
                sequence = repmat(sequence, [resolution^(length(obj.inputs) - m), 1]);
                featureset(m, :) = repmat(sequence(:)', [1, resolution^(m - 1)]);
            end
            featureset = featureset';
            
            v = waitbar(0, 'Acquiring data...');
            for m = 1:size(featureset, 1)
                labels = obj.simulate(featureset(m, :), h);
                obj.examples(end + 1).features = featureset(m, :);
                obj.examples(end).labels = labels;
                waitbar(m/size(featureset, 1));
            end
            close(v);
        end
        
        function check_single(obj, features)
            path(path, 'C:\Program Files\Lumerical\fdtd\api\matlab');
            h = appopen('fdtd');
            obj.simulate(features, h)
        end

        function labels = simulate(obj, features, h)
            code = strcat('load("',char(obj.file_name),'");',...
                'switchtolayout;');
            appevalscript(h, code);

            for n = 1:length(obj.inputs)
                code = strcat('select("',char(obj.inputs(n).structure),'");',...
                    'set("',char(obj.inputs(n).parameter),'", ',num2str(features(n)),');');
                appevalscript(h, code);
            end

            code = strcat('run;');
            appevalscript(h, code);
            
%             code = strcat('runanalysis("Qanalysis");',...
%                 'Q = getresult("Qanalysis", "Q");',...
%                 'labels = [transpose(Q.Q), transpose(Q.lambda)];');
            
            code = strcat('port = getresult("FDTD::ports::port 2", "T");',...
                'T = port.T;',...
                'labels = min(T);');
            appevalscript(h, code);
            labels = abs(appgetvar(h, 'labels')');
        end
        
        function map_examples(obj)
            features = reshape([obj.examples.features], [length(obj.examples(1).features) length(obj.examples)])';
            scatter(features(:, 1), features(:, 2));
            xlim(obj.inputs(1).range);
            ylim(obj.inputs(2).range);
        end
        
        function remove_bad_examples(obj, FOM)
            m = 1;
            while m <= length(obj.examples)
                if abs(min(obj.examples(m).labels)) < abs(FOM)
                    obj.examples(m) = [];
                    m = m - 1;
                end
                m = m + 1;
            end
        end
    end
end
