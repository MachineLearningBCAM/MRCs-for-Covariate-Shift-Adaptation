clear all;
close all;
%{
Input
-----

    The name of dataset file

    If dataset is Blood, BreastCancer, Haberman or Ringnorm
    way to generate covariate shift (based on features or PCA)

    Feature mapping: linear, polinom, polinom3, and RFF

    Deterministic: "true" for Deterministic DWGCS and "false" for DWGCS


Output
------

    Error of DWGCS 

    R(U) of DWGCS 

    D that return the lowest minimax risk R(U) 
%}

% Add functions path
addpath('Auxiliar')
addpath('CovShift_Generation')
addpath('DWGCS')
addpath('Datasets')

%{
For the 20NewsGroups we do not need to generate 
covariate shift since it is intrinsically affect by it.
%}

filename = 'comp_vs_sci.mat';
load(filename)

% Perform Pearson correlation coefficients to select 1000 features
top_features = PCC([X_Train;X_Test],[Y_Train;Y_Test],1000);

X_Train = X_Train(:,top_features);
X_Test = X_Test(:,top_features);
N_tr = size(X_Train,1);
N_te = size(X_Test,1);
X = full(zscore([X_Train;X_Test]));

m = size(X,2);
[~,distance] = knnsearch(X,X,'K',50);
sigma        = mean(distance(:,50));

n_tr = 1000;
idx_tr = randsample(N_tr,n_tr);
x_tr = X_Train(idx_tr,:);
y_tr = Y_Train(idx_tr);

n_te = 1000;
idx_te = randsample(N_te,n_te);
x_te = X_Test(idx_te,:);
y_te = Y_Test(idx_te);
% Define base model parameters

BaseMdl.intercept = false;
BaseMdl.fmapping = 'linear';
BaseMdl.loss = '0-1';
BaseMdl.deterministic = true;
BaseMdl.labels = 2;
BaseMdl.sigma = sigma;
BaseMdl.B = 1000;

% Double-Weighting General Covariate Shift using 0-1-loss

D = 1./(1-(0:0.1:0.9)).^2;
for l=1:length(D)
    for i=1:1
        DWGCS{l} = BaseMdl;
        DWGCS{l}.D = D(l);
    end
    DWGCS{l}     = DWGCS_weights(DWGCS{l},x_tr,x_te);
    DWGCS{l}     = DWGCS_parameters(DWGCS{l},x_tr,y_tr,x_te);
    DWGCS{l}     = DWGCS_learning(DWGCS{l},x_te);
    RU_DWGCS(l) = DWGCS{l}.min_MRC;
end
[RU_bestDWGCS,position] = min(RU_DWGCS);
D_best = D(position);
DWGCS{position} = DWGCS_prediction(DWGCS{position},x_te,y_te);
Error_bestDWGCS = DWGCS{position}.error;
