classdef Data < handle
    properties
        file_name
        inputs = struct('structure', {}, 'parameter', {}, 'range', {})
        outputs = struct('port', {}, 'attribute', {})
        examples = struct('features', {}, 'labels', {})
        wavelengths = [1.53e-6, 1.565e-6]
    end
    methods
        function obj = Data
            obj.file_name = "H:\photonmind-ring\Devices\" + input('Enter the FDTD file name: ', 's');

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

        function get_examples_random(obj, num_sim)
            format long;
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');
            path(path, 'C:\Program Files\Lumerical\DEVICE\api\matlab');
            u = appopen('device');

            featureset = zeros(num_sim, length(obj.inputs));
            for m = 1:num_sim
                for n = 1:length(obj.inputs)
                    featureset(m, n) = obj.inputs(n).range(1) + diff(obj.inputs(n).range)*rand;
                end
            end
            featureset

            for m = 1:size(featureset, 1)
                labels = obj.simulate(featureset(m, :), h, u);
                obj.examples(end + 1).features = featureset(m, :);
                obj.examples(end).labels = labels;
            end
            appclose(u);
        end

        function get_examples_uniform(obj, resolution)     
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');
            path(path, 'C:\Program Files\Lumerical\DEVICE\api\matlab');
            u = appopen('device');

            featureset = zeros(length(obj.inputs), resolution^length(obj.inputs));
            for m = 1:length(obj.inputs)
                sequence = linspace(obj.inputs(m).range(1), obj.inputs(m).range(2), resolution);
                sequence = repmat(sequence, [resolution^(length(obj.inputs) - m), 1]);
                featureset(m, :) = repmat(sequence(:)', [1, resolution^(m - 1)]);
            end
            featureset = featureset';

            for m = 1:size(featureset, 1)
                labels = obj.simulate(featureset(m, :), h, u);
                obj.examples(end + 1).features = featureset(m, :);
                obj.examples(end).labels = labels;
            end
            appclose(u);
        end
        
        function get_single(obj, features)
            path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
            h = appopen('fdtd');
            path(path, 'C:\Program Files\Lumerical\DEVICE\api\matlab');
            u = appopen('device');
            obj.simulate(features, h, u)
        end

        function labels = simulate(obj, features, h, u)
            format long;
            obj.get_charge_profile(features(3), features(4), features(2), u);
            
            code = strcat('load("',char(obj.file_name),'");',...
                'switchtolayout;',...
                'select("np density");',...
                'importdataset("charge.mat");',...
                'set("V_p_contact_index", 1);');
            appevalscript(h, code);

%             for n = 1:length(obj.inputs)
            for n = 1:2
                code = strcat('select("',char(obj.inputs(n).structure),'");',...
                    'set("',char(obj.inputs(n).parameter),'", ',num2str(features(n)),');');
                appevalscript(h, code);
            end

            code = strcat('run;');
            appevalscript(h, code);

            labels = [];
            for n = 1:length(obj.outputs)
                code = strcat('port = getresult("FDTD::ports::',char(obj.outputs(n).port),'","T");',...
                    'T = port.T;',...
                    'lam = port.lambda;',...
                    'Q = getresult("lowQanalysis", "Q");',...
                    'closeall;',...
                    'Q = Q.Q;');
                appevalscript(h, code);
                T = appgetvar(h, char('T'))';
                Q = appgetvar(h, char('Q'))';
                labels = obj.find_ring_spectrum_params(T, Q);
            end
            
            code = strcat('switchtolayout;',...
                'select("np density");',...
                'set("V_p_contact_index", 3);',...
                'run;');
            appevalscript(h, code);
            
            code = strcat('port = getresult("FDTD::ports::',char(obj.outputs(n).port),'","T");',...
                'T = port.T;',...
                'lam = port.lambda;');
            appevalscript(h, code);
            T = appgetvar(h, char('T'))';
            lam = linspace(obj.wavelengths(2), obj.wavelengths(1), length(T));
            [pks, locs] = findpeaks(1 - T, 'MinPeakProminence', 0.1);
            locs = lam(locs);
            if length(labels) == 6
                del_lam = abs(labels(5) - locs(1));
            else
                del_lam = 0;
            end
            labels = cat(2, labels, del_lam);
        end
        
        function remove_edge_minimums(obj)
            ind = find(strcmp({obj.outputs.attribute}, 'lam_T_min') == 1);
            m = 1;
            while m <= length(obj.examples)
                if obj.examples(m).labels(ind) == obj.wavelengths(1)...
                    || obj.examples(m).labels(ind) == obj.wavelengths(2)
                    obj.examples(m) = [];
                    m = m - 1;
                end
                m = m + 1;
            end
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
        
        function remove_unmatched_spectrums(obj)
            m = 1;
            while m <= length(obj.examples)
                if length(obj.examples(m).labels) ~= 7
                    obj.examples(m) = [];
                end
                m = m + 1;
            end
        end

        function y = find_ring_spectrum_params(obj, T, Q)
            Q = linspace(Q(1), Q(2), length(T));
            lam = linspace(obj.wavelengths(2), obj.wavelengths(1), length(T));
            [pks, locs] = findpeaks(1 - T, 'MinPeakProminence', 0.1);
            pks = 1 - pks;
            qs = cat(2, pks, Q(locs));
            y = cat(2, qs, lam(locs));
        end
        
        function get_charge_profile(~, ND, NA, R, u)  
            NA = -NA;
            wg_width = 0.5e-6;
            P = 401;
            
            x = linspace(-10, 10, P)*1e-6;
            y = linspace(-(R + wg_width/2)*1e6, (R + wg_width/2)*1e6, P)*1e-6;
            z = linspace(0, 0.22, 2)*1e-6; %#ok
            N = zeros(P,P,2);
            N(1:P,1:P,1:2) = NA;
            
            for i = 1:P
                for j = 1:P
                    if sqrt(x(i)^2 + y(j)^2) < R
                        N(i,j,1:2) = ND;
                    end
                end
            end
            save('Devices/doping','x','y','z','N');
            
%             path(path, 'C:\Program Files\Lumerical\DEVICE\api\matlab');
%             h = appopen('device');
            
            code = strcat('load("ring_modulator");',...
                'switchtolayout;');
            appevalscript(u, code);
            
            code = strcat('matlabload("doping");',...
                'doping = rectilineardataset(x,y,z);',...
                'doping.addparameter("a",0);',...
                'doping.addattribute("N",N);',...
                'select("CHARGE::doping");',...
                'importdataset(doping);',...
                'run;');
            appevalscript(u, code);
        end
    end
end
