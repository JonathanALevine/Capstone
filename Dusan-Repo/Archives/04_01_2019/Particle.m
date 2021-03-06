classdef Particle < handle
    properties
        inputs = struct('structure', {}, 'parameter', {}, 'range', {})
        position
        best_position
        FOM
        best_FOM = 0
        velocity
    end
    methods
        function obj = Particle
%             for the lack of a better method for now, manually enter
%             inputs here
            obj.inputs(end + 1) = struct('structure', {'gc'}, 'parameter',...
                {'etch depth'}, 'range', {[0.02e-6, 0.2e-6]});
            obj.inputs(end + 1) = struct('structure', {'gc'}, 'parameter',...
                {'duty cycle'}, 'range', {[0.1, 0.9]});
            obj.inputs(end + 1) = struct('structure', {'gc'}, 'parameter',...
                {'pitch'}, 'range', {[0.2e-6, 1e-6]});
            obj.inputs(end + 1) = struct('structure', {'fiber'}, 'parameter',...
                {'theta0'}, 'range', {[13, 25]});
%             obj.inputs(end + 1) = struct('structure', {'CHARGE::p_top_a'},...
%                 'parameter', {'range'}, 'range', {[0.06e-6, 0.17e-6]});
%             obj.inputs(end + 1) = struct('structure', {'CHARGE::p_top_b'},...
%                 'parameter', {'range'}, 'range', {[0.06e-6, 0.17e-6]});
%             obj.inputs(end + 1) = struct('structure', {'CHARGE::p_top_a'},...
%                 'parameter', {'straggle'}, 'range', {[0.01e-6, 0.08e-6]});
%             obj.inputs(end + 1) = struct('structure', {'CHARGE::p_top_b'},...
%                 'parameter', {'straggle'}, 'range', {[0.01e-6, 0.08e-6]});
%             obj.inputs(end + 1) = struct('structure', {'zipper'},...
%                 'parameter', {'pitch'}, 'range', {[0.2e-6, 0.7e-6]});
            
            for n = 1:length(obj.inputs)
                obj.position(end + 1) = obj.inputs(n).range(1)...
                    + (obj.inputs(n).range(2) - obj.inputs(n).range(1))*rand;
            end
            obj.best_position = obj.position;
            obj.velocity = zeros(size(obj.position));
        end
    end
end
