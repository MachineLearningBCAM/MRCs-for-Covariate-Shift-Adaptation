import numpy as np
from scipy.spatial import distance
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Select_20News import Select_20News
from Pearson_Corr_Coeffs import PCC

def main():
    # For Mac
    path = '/Users/jsegovia/Python/MRCs-for-Covariate-Shift-Adaptation/'
    add_paths = [
        '/Users/jsegovia/cvx'
        'Datasets_20News/',
        'Auxiliary_Functions/',
        'Cov_Shift_Gen_Functions/',
        'DWGCS/',
    ]

    # Add paths
    import sys
    sys.path.append(path)
    for add_path in add_paths:
        sys.path.append(add_path)

    from DWGCS import DWGCS   

    X_Train = np.genfromtxt('Datasets_20News/comp_vs_sci_X_Train.csv', delimiter=',')
    Y_Train = np.genfromtxt('Datasets_20News/comp_vs_sci_Y_Train.csv', delimiter=',')       
    X_Test = np.genfromtxt('Datasets_20News/comp_vs_sci_X_Test.csv', delimiter=',')
    Y_Test = np.genfromtxt('Datasets_20News/comp_vs_sci_Y_Test.csv', delimiter=',')

    # If dataset is not processed using Pearson correlation coefficient 
    # use function PCC (just for non sparse data)
    # D = 1000 #number of features we want to select
    # [X_Train,X_Test]=PCC(X_Train, X_Test, Y_Train, Y_Test, D)

    N_tr = X_Train.shape[0]
    n_tr = 1000
    N_te = X_Test.shape[0]
    n_te = 1000
    X = np.concatenate((X_Train, X_Test))
    X = StandardScaler().fit_transform(X)
    # X_Train = X[:N_tr,:]
    # X_Test  = X[N_tr:,:]

    m = X.shape[1]
    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    sigma_ = np.mean(distances[:, 49])

    class BaseMdl:
        def __init__(self,intercep,deterministic,feature_map,labels,lambda0,loss,sigma_):
            self.intercep = intercep
            self.deterministic = deterministic
            self.feature_map = feature_map
            self.labels = labels
            self.lambda0 = lambda0 
            self.loss = loss
            self.sigma_ = sigma_
        
    idx_tr = resample(range(N_tr), n_samples=n_tr)
    xtr = X_Train[idx_tr, :]
    ytr = Y_Train[idx_tr].astype(int).reshape(-1, 1)

    idx_te = resample(range(N_te), n_samples=n_te)
    xte = X_Test[idx_te, :]
    yte = Y_Test[idx_te].astype(int).reshape(-1, 1)


    # DWGCS 0-1-loss
    D = 1.0 / np.square(1.0 - (np.arange(0.0, 1.0, 0.1)))
    RU_Dwgcs = np.zeros(len(D))

    Dwgcs =[]
    for l in range(len(D)):
        MdlAux = BaseMdl(False,True,'linear',2,0,'0-1',sigma_)
        MdlAux.D = D[l]
        Dwgcs.append(MdlAux)
        Dwgcs[l] = DWGCS.DWKMM(Dwgcs[l],xtr,xte)
        Dwgcs[l] = DWGCS.parameters(Dwgcs[l],xtr,ytr,xte)
        Dwgcs[l] = DWGCS.learning(Dwgcs[l],xte)
        RU_Dwgcs[l] = Dwgcs[l].RU

    RU_best_Dwgcs = np.min(RU_Dwgcs)
    position = np.argmin(RU_Dwgcs)
    Dwgcs[position] = DWGCS.prediction(Dwgcs[position],xte,yte)
    error_best_Dwgcs = Dwgcs[position].error




if __name__ == '__main__':
    main()