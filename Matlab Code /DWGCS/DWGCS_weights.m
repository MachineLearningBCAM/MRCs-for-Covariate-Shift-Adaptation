function Mdl = DWGCS_weights(Mdl,x_tr,x_ts)

n = size(x_tr,1);
t = size(x_ts,1);
x = [x_tr;x_ts];
epsilon = 1-1/sqrt(n);

K = zeros(n+t,n+t);
for i = 1:n+t
    K(i,i) = 1/2;
    for j = i+1:n+t
        K(i,j) = exp(-norm(x(i,:)-x(j,:))^2/(2*Mdl.sigma^2));
    end
end
K = K+K';

if Mdl.D == 1

    beta=[];
    alpha=ones(t,1);
    cvx_begin quiet
    variables beta(n,1)

    minimize( [beta/n;-alpha/t]'*K*[beta/n;-alpha/t] )
    subject to
    beta                  >= zeros(n,1);
    beta                  <= (Mdl.B/sqrt(Mdl.D))*ones(n,1);
    abs(sum(beta)/n-sum(alpha)/t) <= epsilon;
    cvx_end

else

    beta=[];
    alpha=[];
    cvx_begin quiet
    variables beta(n,1) alpha(t,1)

    minimize( [beta/n;-alpha/t]'*K*[beta/n;-alpha/t] )
    subject to
    beta                  >= zeros(n,1);
    beta                  <= (Mdl.B/sqrt(Mdl.D))*ones(n,1);
    alpha                 >= zeros(t,1);
    alpha                 <= ones(t,1);
    abs(sum(beta)/n-sum(alpha)/t) <= epsilon;
    norm(alpha-ones(t,1)) <= (1-1/sqrt(Mdl.D))*sqrt(t);
    cvx_end

end

Mdl.min_KMM = cvx_optval;

Mdl.beta  = abs( beta );
Mdl.alpha = abs( alpha );
end