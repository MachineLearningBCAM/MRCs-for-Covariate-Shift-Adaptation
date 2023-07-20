function canonical_vector = e(y,n_clases)

for i = 1:length(y)
    canonical_vector(i,:) = [zeros(1,y(i)-1),1,zeros(1,n_clases-y(i))];
end

end

