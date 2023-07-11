import numpy as np
from scipy.spatial import distance
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from Select_20News import Select_20News

def main():
    # For Mac
    path = '/Users/jsegovia/cvx'
    add_paths = [
        '../Datasets/',
        '../Auxiliary_Functions/',
        '../Cov_Shift_Gen_Functions/',
        '../Reweighted',
        '../Robust/',
        '../DWGCS/',
    ]

    # Add paths
    import sys
    # sys.path.append(path)
    # for add_path in add_paths:
    # sys.path.append(add_path)
    sys.path.insert(1, '../Reweighted')
    

    from Reweighted import Reweighted
    from Robust import Robust
    from DWGCS import DWGCS

    Table_errors = np.zeros((10,7))
    Table_errorD_01 = np.zeros((10,10))
    Table_errorD_log = np.zeros((10,10))

    for i in range(5):

        [Train_Set,Test_Set] = Select_20News(i)
        d = Train_Set.shape[0]
        X_Train = Train_Set[:,d-1]
        X_Test = Train_Set[:,-1]
        Y_Train = Test_Set[:,d-1]
        Y_Test = Test_Set[:,-1]

        N_tr = X_Train.shape[0]
        n_tr = 1000
        N_te = X_Test.shape[0]
        n_te = 1000
        X = np.concatenate((X_Train, X_Test))
        X = StandardScaler().fit_transform(X)

        m = X.shape[1]
        nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)
        sigma_ = np.mean(distances[:, 49])

        class Mdl:
            def __init__(self,intercep,deterministic,feature_map,labels,lambda0,loss,sigma_):
                self.intercep = intercep
                self.deterministic = deterministic
                self.feature_map = feature_map
                self.labels = labels
                self.lambda0 = lambda0 
                self.loss = loss
                self.sigma_ = sigma_

        Base_Model = Mdl(False,True,'linear',2,0,'log',sigma_)
    
        RU_IW = np.zeros(len(rep))
        error_IW = np.zeros(len(rep))
        RU_Flat = np.zeros(len(rep))
        error_Flat = np.zeros(len(rep))
        RU_Rulsif = np.zeros(len(rep))
        error_Rulsif = np.zeros(len(rep))
        RU_Rob = np.zeros(len(rep))
        error_Rob = np.zeros(len(rep))
        RU_Kmm = np.zeros(len(rep))
        error_Kmm = np.zeros(len(rep))
        RU_best_Dwgcs_01 = np.zeros(len(rep))
        error_best_Dwgcs_01 = np.zeros(len(rep))
        RU_best_Dwgcs_log = np.zeros(len(rep))
        error_best_Dwgcs_log = np.zeros(len(rep))
        
        for rep in range(100):
            idx_tr = resample(range(N_tr), n_samples=n_tr)
            xtr = X_Train[idx_tr, :]
            ytr = Y_Train[idx_tr]

            idx_te = resample(range(N_te), n_samples=n_te)
            xte = X_Test[idx_te, :]
            yte = Y_Test[idx_te]

            # Reweighted
            IW = Mdl(False,True,'linear',2,0,'log',sigma_)
            IW = Reweighted.LREIW(IW,xtr,xte)
            IW = Reweighted.parameters(IW,xtr,ytr)
            IW = Reweighted.learning(IW,xtr)
            IW = Reweighted.prediction(IW,xte,yte)
            RU_IW[rep] = IW.RU
            error_IW[rep] = IW.error

            # Reweighted Flattening
            Flat = Mdl(False,True,'linear',2,0,'log',sigma_)
            Flat.beta_ = np.sqrt(IW.beta_);
            Flat = Reweighted.parameters(Flat,xtr,ytr)
            Flat = Reweighted.learning(Flat,xtr)
            Flat = Reweighted.prediction(Flat,xte,yte)
            RU_Flat[rep] = Flat.RU
            error_Flat[rep] = Flat.error

            # Reweighted RuLSIF
            Rulsif = Mdl(False,True,'linear',2,0,'log',sigma_)
            Rulsif = Reweighted.RuLSIF(Rulsif,xtr,xte)
            Rulsif = Reweighted.parameters(Rulsif,xtr,ytr)
            Rulsif = Reweighted.learning(Rulsif,xtr)
            Rulsif = Reweighted.prediction(Rulsif,xte,yte)
            RU_Rulsif[rep] = Rulsif.RU
            error_Rulsif[rep] = Rulsif.error

            # Robust
            Rob = Mdl(False,True,'linear',2,0,'log',sigma_)
            Rob = Robust.LREIW(Rob,xtr,xte)
            Rob = Robust.parameters(Rob,xtr,ytr)
            Rob = Robust.learning(Rob,xtr)
            Rob = Robust.prediction(Rob,xte,yte)
            RU_Rob[rep] = Rob.RU
            error_Rob[rep] = Rob.error

            # KMM
            Kmm = Mdl(False,True,'linear',2,0,'log',sigma_)
            Kmm = Reweighted.KMM(Kmm,xtr,xte)
            Kmm = Reweighted.parameters(Kmm,xtr,ytr)
            Kmm = Reweighted.learning(Kmm,xtr)
            Kmm = Reweighted.prediction(Kmm,xte,yte)
            RU_Kmm[rep] = Kmm.RU
            error_Kmm[rep] = Kmm.error

            # DKMM 0-1-loss
            D = 1.0 / np.square(1.0 - (np.arange(0.0, 1.0, 0.1)))
            RU_Dwgcs_01 = np.zeros(len(D))
            error_Dwgcs_01 = np.zeros(len(D))

            for l in range(len(D)):
                Dwgcs_01 = Mdl(False,True,'linear',2,0,'0-1',sigma_)
                Dwgcs_01.D = D[l]
                Dwgcs_01 = DWGCS.DWKMM(Dwgcs_01,xtr,xte)
                Dwgcs_01 = DWGCS.parameters(Dwgcs_01,xtr,ytr,xte)
                Dwgcs_01 = DWGCS.learning(Dwgcs_01,xte)
                Dwgcs_01 = DWGCS.prediction(Dwgcs_01,xte,yte)
                RU_Dwgcs_01[l] = Dwgcs_01[l].RU
                error_Dwgcs_01[l] = Dwgcs_01[l].error

            RU_best_Dwgcs_01[rep] = np.min(RU_Dwgcs_01)
            position = np.argmin(RU_Dwgcs_01)
            error_best_Dwgcs_01[rep] = error_Dwgcs_01[position]

            # DKMM log-loss
            RU_Dwgcs_log = np.zeros(len(D))
            error_Dwgcs_log = np.zeros(len(D))

            for l in range(len(D)):
                Dwgcs_log = Mdl(False,True,'linear',2,0,'log',sigma_)
                Dwgcs_log.D = D[l]
                Dwgcs_log = DWGCS.DWKMM(Dwgcs_log,xtr,xte)
                Dwgcs_log = DWGCS.parameters(Dwgcs_log,xtr,ytr,xte)
                Dwgcs_log = DWGCS.learning(Dwgcs_log,xte)
                Dwgcs_log = DWGCS.prediction(Dwgcs_log,xte,yte)
                RU_Dwgcs_log[l] = Dwgcs_log[l].RU
                error_Dwgcs_log[l] = Dwgcs_log[l].error

            RU_best_Dwgcs_log[rep] = np.min(RU_Dwgcs_log)
            position = np.argmin(RU_Dwgcs_log)
            error_best_Dwgcs_log[rep] = error_Dwgcs_log[position]

            # Grouping Results of Methods

            Errors = ([np.mean(error_IW),np.mean(error_Flat),np.mean(error_Rulsif), +
                    np.mean(error_Rob),np.mean(error_Kmm), +
                    np.mean(error_best_Dwgcs_01),np.mean(error_best_Dwgcs_log)])
            All_errors = np.concatenate(Errors, axis=1)

            Errors_std = ([np.std(error_IW),np.std(error_Flat),np.std(error_Rulsif), +
                    np.std(error_Rob),np.std(error_Kmm), +
                    np.std(error_best_Dwgcs_01),np.std(error_best_Dwgcs_log)])
            All_errors_std = np.concatenate(Errors_std, axis=1)

            Mins = ([np.mean(RU_IW),np.mean(RU_Flat),np.mean(RU_Rulsif), +
                    np.mean(RU_Rob),np.mean(RU_Kmm), +
                    np.mean(RU_best_Dwgcs_01),np.mean(RU_best_Dwgcs_log)])
            All_mins = np.concatenate(Mins, axis=1)

            all_errorD_01 = np.zeros((100, len(D)))
            all_minD_01 = np.zeros((100, len(D)))
            all_errorD_log = np.zeros((100, len(D)))
            all_minD_log = np.zeros((100, len(D)))

            for l in range(len(D)):
                all_errorD_01[rep, l] = Dwgcs_01[l].error
                all_minD_01[rep, l] = Dwgcs_01[l].RU
                all_errorD_log[rep, l] = Dwgcs_log[l].error
                all_minD_log[rep, l] = Dwgcs_log[l].RU

            # Generate the tables

            Table_errors[2*i:2*i+1,:] = np.concatenate((Errors, Errors_std),axis=0)
            Table_errorD_01[2*i:2*i+1,:] = np.concatenate((np.mean(all_errorD_01, axis=0), np.std(all_errorD_01, axis=0)))
            Table_errorD_log[2*i:2*i+1,:] = np.concatenate((np.mean(all_errorD_log, axis=0), np.std(all_errorD_log, axis=0)))


if __name__ == '__main__':
    main()