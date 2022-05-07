data = pn_v;
a = data.examples(1).labels;
for n = 2:length(data.examples)
    a(end + 1) = data.examples(n).labels;
end
