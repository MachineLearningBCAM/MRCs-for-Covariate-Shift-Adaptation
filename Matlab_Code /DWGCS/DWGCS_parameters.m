function Mdl = DWGCS_parameters(Mdl,x_tr,y_tr,x_ts)

auxtau = [];

n = size(x_tr,1);
t = size(x_ts,1);

for i=1:n
    auxtau = [auxtau;Mdl.beta(i)*phi(Mdl,x_tr(i,:),y_tr(i))];
end

Mdl.tau = sum(auxtau)/n;

delta = 1e-6;

cvx_begin quiet
variables lambda(size(Mdl.tau)) p(t,Mdl.labels)
aux = [];
for i = 1:t
    for j = 1:Mdl.labels
        aux = [aux;p(i,j)*Mdl.alpha(i)*phi(Mdl,x_ts(i,:),j)];
    end
end
minimize(ones(1,length(Mdl.tau))*lambda')
subject to
Mdl.tau-lambda+delta <= sum(aux);
sum(aux)             <= Mdl.tau+lambda-delta;
zeros(size(Mdl.tau)) <= lambda;
sum(p,2)             == ones(t,1)/t;
p                    >= 0;
cvx_end

Mdl.lambda = lambda;

for i = 1:length(Mdl.lambda)
    if Mdl.lambda(i) <= 0
        Mdl.lambda(i) = 0;
    end
end

end
