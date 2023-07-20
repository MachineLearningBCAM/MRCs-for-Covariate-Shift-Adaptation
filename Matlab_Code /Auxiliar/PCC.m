function top_features=PCC(X,Y,D)

[~, m] = size(X);

pearson_coeffs = zeros(1,m);
for i = 1:m
    pearson_coeffs(i) = corr(X(:,i), Y,'rows','complete');
end
pearson_coeffs=pearson_coeffs(~isnan(pearson_coeffs));
[~, sorted_indices] = sort(abs(pearson_coeffs), 'descend');



top_features = sorted_indices(1:D);

end