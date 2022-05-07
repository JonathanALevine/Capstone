gc_split_disc.examples = [];

lam = linspace(gc_split_disc.wavelengths(1), gc_split_disc.wavelengths(2), 200);
ix = 1;
for n = 1:length(gc_split.examples)
    p1 = gc_split.examples(n).labels(1:200);
    p2 = gc_split.examples(n).labels(201:400);
    [T1, loc1] = min(p1);
    [T2, loc2] = max(p2);
    if loc1 ~= 1 && loc1 ~= 200 && loc2 ~= 1 && loc2 ~= 200
        gc_split_disc.examples(ix).features = gc_split.examples(n).features;
        gc_split_disc.examples(ix).labels = [T1, lam(loc1), T2, lam(loc2)];
        ix = ix + 1;
    end
end

gc_split_disc %#ok

clear lam n ans T1 T2 p1 p2 loc1 loc2 ix