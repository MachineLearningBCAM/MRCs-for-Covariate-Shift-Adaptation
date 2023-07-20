function Mdl = DWGCS_learning(Mdl,x)

mu = [];
t  = size(x,1);

if strcmp(Mdl.loss,'0-1')

    v  = zeros(2^Mdl.labels-1,1);

    pset = powerset(Mdl.labels);
    for i = 1:t
        for j = 1:(2^Mdl.labels-1)
            M{i,1}(j,:) = sum(Mdl.alpha(i)*phi(Mdl,x(i,:),pset{j}),1)/size(pset{j},1);
        end
    end

    for j = 1:(2^Mdl.labels-1)
        v(j,1) = 1/size(pset{j},1);
    end
    v = repmat(v,1,t);

    cvx_begin quiet
    variable mu(size(Mdl.tau,2),1)
    minimize( -Mdl.tau*mu+sum(ones(1,t)+max(reshape(cell2mat(M)*mu,2^Mdl.labels-1,t)-v))/t+Mdl.lambda*abs(mu)  )
    cvx_end
end

if strcmp(Mdl.loss,'log')

    for i=1:t
        M{i} = Mdl.alpha(i)*phi(Mdl,x(i,:),(1:Mdl.labels));
    end
    cvx_begin quiet
    variable mu(size(Mdl.tau,2),1)
    minimize( -Mdl.tau*mu+phi_mu(M,mu)/t+Mdl.lambda*abs(mu) )
    cvx_end
end

Mdl.mu       = mu;
Mdl.min_MRC = cvx_optval;

end

function value = phi_mu(M,mu)
value=0;
for k=1:length(M)
    value = value+log_sum_exp(M{k}*mu);
end
end