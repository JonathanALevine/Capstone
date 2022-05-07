count = 0;
m = 1;
while m <= length(gc_swg_disc.examples)
    if gc_swg_disc.examples(m).labels(1) > -0.05
        count = count + 1;
        gc_swg_disc.examples(m) = [];
        m = m - 1;
    end
    m = m + 1;
end

gc_swg_disc %#ok
% count %#ok
clear count m