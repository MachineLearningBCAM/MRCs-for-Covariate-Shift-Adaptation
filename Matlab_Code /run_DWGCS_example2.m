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
