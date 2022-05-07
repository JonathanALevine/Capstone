classdef Source < handle
    properties
        amplitude
        wavelength
        broadband = false
    end
    methods
        function obj = Source(amplitude, wavelength)
            obj.amplitude = amplitude;
            obj.wavelength = wavelength;
            if length(wavelength) > 1, obj.broadband = true; end
        end
    end
end
