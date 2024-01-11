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

% Load the dataset and compute sigma

filename = 'Blood.mat';
load(filename)
dataset = blood_normalize;

X = dataset(:,1:(end-1));
m = size(X,2);
[~,distance] = knnsearch(X,X,'K',50);
sigma        = mean(distance(:,50));

% Covariate Shift Generation:

%{
 In this case we generate covariate shift based on the first
 feature. We select 0.7 and 0.3 probabilities.
%}

feature = 1;
[Train_Set,Test_Set] = CSG_Features(dataset,feature,0.7,0.3);

x_tr = Train_Set(:,1:(end-1));
y_tr = Train_Set(:,end);

x_te = Test_Set(:,1:(end-1));
y_te = Test_Set(:,end);

%{ 
In this case we generate covariate shift based on the first
feature for the BreastCancer dataset. We select 0.7 and 0.3 probabilities.

n = 100;
t = 100;
[Train_Set,Test_Set] = CSG_Features_BreastCancer(dataset_original,dataset,feature,0.3,0.7,n,t);

x_tr = Train_Set(:,1:(end-1));
y_tr = Train_Set(:,end);

x_te = Test_Set(:,1:(end-1));
y_te = Test_Set(:,end);
%}

%{ 
In this case we generate covariate shift based on the first
 principal component. We select 0.7 and 0.3 probabilities.

[Train_Set,Test_Set,n,t] = CSG_PCA(dataset,0.7,0.3);

x_tr = Train_Set(:,1:(end-1));
y_tr = Train_Set(:,end);

x_te = Test_Set(:,1:(end-1));
y_te = Test_Set(:,end);
%}

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
