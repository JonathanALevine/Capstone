gc_swg_disc.examples = [];

lam = linspace(gc_swg_disc.wavelengths(1), gc_swg_disc.wavelengths(2), 500);
ix = 1;
for n = 1:length(gc_swg_20deg.examples)
    [T, loc] = min(gc_swg_20deg.examples(n).labels);
    if loc ~= 1 && loc ~= 500
        gc_swg_disc.examples(ix).features = gc_swg_20deg.examples(n).features;
        gc_swg_disc.examples(ix).labels = [T, lam(loc)];
        ix = ix + 1;
    end
end

gc_swg_disc %#ok

clear lam loc n ans T ix
