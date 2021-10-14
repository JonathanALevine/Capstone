path(path, 'C:\Program Files\Lumerical\FDTD\api\matlab');
h = appopen('fdtd');
code = strcat('load("H:\photonmind-master\Devices\grating_coupler_swg_split.fsp");',...
    'runsweep("optimization1");');
appevalscript(h, code);