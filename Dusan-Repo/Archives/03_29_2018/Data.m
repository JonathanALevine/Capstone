classdef Data < handle
    properties
        file
        inputs = struct('structure', {}, 'parameter', {}, 'range', {});
        outputs = struct('port', {}, 'attribute', {});
        examples = struct('features', {}, 'labels', {});
    end
    methods
        function obj = Data
            obj.file = input('Enter the FDTD file path: ', 's');
            
            while input('Would you like to add an input? Y/N: ', 's') ~= 'n'
                obj.inputs(end + 1).structure = input('Enter the structure name: ', 's');
                obj.inputs(end).parameter = input('Enter the parameter name: ', 's');
                user_range = input('Enter the range as a 1x2 matrix: ');
                obj.inputs(end).range = [(user_range(1) - 0.1*diff(user_range)),...
                    (user_range(2) + 0.1*diff(user_range))];
            end
            
            while input('Would you like to add an output? Y/N: ', 's') ~= 'n'
                obj.outputs(end + 1).port = input('Enter the port number: ', 's');
                obj.outputs(end).attribute = input('Enter the attribute: ', 's');
            end
        end
        
        function get_examples(obj, num_sim)
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');
            
            featureset = obj.get_featureset(num_sim);
            
            for i = 1:size(featureset, 1)
                labels = obj.simulate(featureset(i, :), h);
                obj.examples(end + 1) = struct('features', {featureset(i, :)}, 'labels', {labels});
            end
        end
        
        function featureset = get_featureset(obj, num_sim)
            featureset = zeros(num_sim, length(obj.inputs));
            for i = 1:num_sim            
                for j = 1:length(obj.inputs)
                    featureset(i, j) = obj.inputs(j).range(1) + diff(obj.inputs(j).range)*rand;
                end
            end
        end
        
        function labels = simulate(obj, features, h)
            code = strcat('load("',char(obj.file),'");',...
                'switchtolayout;');
            appevalscript(h, code);
            
            for i = 1:length(obj.inputs)
                code = strcat('select("',char(obj.inputs(i).structure),'");',...
                    'set("',char(obj.inputs(i).parameter),'", ',num2str(features(i)),');');
                appevalscript(h, code);
            end
            
            code = strcat('run;');
            appevalscript(h, code);
            
            labels = zeros(1, length(obj.outputs));
            for i = 1:length(obj.outputs)
                code = strcat('port = getresult("FDTD::ports::port',char(obj.outputs(i).port),'","T");',...
                    'T = port.T;',...
                    'lam = port.lambda;',...
                    'T_min = min(port.T);',...
                    'T_max = max(port.T);',...
                    'lam_T_min = port.lambda(find(port.T, min(port.T)));',...
                    'lam_T_max = port.lambda(find(port.T, max(port.T)));');
                appevalscript(h, code);
                labels(i) = appgetvar(h, char(obj.outputs(i).attribute));
            end
        end
    end
end
