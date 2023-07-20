function [Tr_Set,Te_Set] = CSG_Features_BreastCancer(dataset,dataset_normalize,feature,p1,p2,n,t)



Train_Set = [];
Test_Set  = [];

for i=1:size(dataset,1)
    if dataset(i,feature)<=5
        if rand<=p1
            Train_Set = [Train_Set;dataset_normalize(i,:)];
        else
            Test_Set = [Test_Set;dataset_normalize(i,:)];
        end
    end
    if dataset(i,feature)>5
        if rand<=p2
            Train_Set = [Train_Set;dataset_normalize(i,:)];
        else
            Test_Set = [Test_Set;dataset_normalize(i,:)];
        end
    end
end

N = size(Train_Set,1);
T = size(Test_Set,1);
idx_train = randperm(N,n);
idx_test  = randperm(T,t);

Tr_Set = Train_Set(idx_train,:);
Te_Set = Test_Set(idx_test,:);
end