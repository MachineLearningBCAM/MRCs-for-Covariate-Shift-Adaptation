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

    Error of DWGCS for both 0-1-loss (Error_bestDWGCS_01)
    and log-loss (Error_bestDWGCS_log)

    R_U of DWGCS for both 0-1-loss (RU_bestDWGCS_01)
    and log-loss (RU_bestDWGCS_log)

    D that return the lowest minimax risk R(U) both 
    0-1-loss (D_01) and log-loss (D_log)
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
BaseMdl.deterministic = true;
BaseMdl.labels = 2;
BaseMdl.sigma = sigma;
BaseMdl.B = 1000;

% Double-Weighting General Covariate Shift using 0-1-loss

D = 1./(1-(0:0.1:0.9)).^2;
for l=1:length(D)
    for i=1:1
        DWGCS_01{l} = BaseMdl;
        DWGCS_01{l}.loss = '0-1';
        DWGCS_01{l}.D = D(l);
    end
    DWGCS_01{l}     = DWKMM(DWGCS_01{l},x_tr,x_te);
    DWGCS_01{l}     = DWMRC_parameters(DWGCS_01{l},x_tr,y_tr,x_te);
    DWGCS_01{l}     = DWMRC_learning(DWGCS_01{l},x_te);
    RU_DWGCS_01(l) = DWGCS_01{l}.min_MRC;
end
[RU_bestDWGCS_01,position] = min(RU_DWGCS_01);
D_01 = D(position);
DWGCS_01{position} = DWMRC_prediction(DWGCS_01{position},x_te,y_te);
Error_bestDWGCS_01 = DWGCS_01{position}.error;

% Double-Weighting General Covariate Shift using log-loss

D = 1./(1-(0:0.1:0.9)).^2;
for l=1:length(D)
    for i=1:1
        DWGCS_log{l} = BaseMdl;
        DWGCS_log{l}.loss = 'log';
        DWGCS_log{l}.D = D(l);
    end
    DWGCS_log{l}     = DWKMM(DWGCS_log{l},x_tr,x_te);
    DWGCS_log{l}     = DWMRC_parameters(DWGCS_log{l},x_tr,y_tr,x_te);
    DWGCS_log{l}     = DWMRC_learning(DWGCS_log{l},x_te);
    RU_DWGCS_log(l) = DWGCS_log{l}.min_MRC;
end
[RU_bestDWGCS_log,position] = min(RU_DWGCS_log);
D_log = D(position);
DWGCS_log{position}     = DWMRC_prediction(DWGCS_log{position},x_te,y_te);
Error_bestDWGCS_log = DWGCS_log{position}.error;
