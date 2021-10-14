maxFOM = 0;
maxdel = 0;
for n = 1:length(pn.examples)
    if pn.examples(n).labels(2)/pn.examples(n).labels(1) > maxFOM
        maxFOM = pn.examples(n).labels(2)/pn.examples(n).labels(1)
        n
    end
    if pn.examples(n).labels(2) > maxdel
        maxdel = pn.examples(n).labels(2)
        n
    end
end