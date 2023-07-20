function [Tr_Set,Te_Set] = CSG_PCA(dat,p1,p2,n,t)

% Covariate Shift Generation using the median 
% of the first principal component

datx = dat(:,1:end-1);
FV = pca(datx);
pc = FV(:,1);
dat_pca = datx*pc;
m = median(dat_pca);

Train_Set = [];
Test_Set  = [];

for i=1:size(datx,1)
    if dat_pca(i) >= m
        if rand <= p1
            Train_Set = [Train_Set;dat(i,:)];
        else
            Test_Set = [Test_Set;dat(i,:)];
        end
    end
    if dat_pca(i) < m
        if rand <= p2
            Train_Set = [Train_Set;dat(i,:)];
        else
            Test_Set = [Test_Set;dat(i,:)];
        end
    end
end

N = size(Train_Set,1);
T = size(Test_Set,1);
if exist('n','var')
    if n > N
        n = N;
    end
    if t > T
        t = T;
    end
else
    if N>=1000
        n = 500;
    else
        n = round (N/2);
    end
    if T>=1000
        t = 500;
    else
        t = round (T/2);
    end
end
idx_train = randperm(N,n);
idx_test  = randperm(T,t);

Tr_Set = Train_Set(idx_train,:);
Te_Set = Test_Set(idx_test,:);

end