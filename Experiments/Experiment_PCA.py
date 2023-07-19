import numpy as np
from scipy.spatial import distance
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from Select_Dataset import Select_Dataset
from CovShiftGen import CovShiftGen

def main():
    # For Mac
    path = '/Users/jsegovia/Python/MRCs-for-Covariate-Shift-Adaptation/'
    add_paths = [
        '/Users/jsegovia/cvx'
        'Datasets/',
        'Auxiliary_Functions/',
        'Cov_Shift_Gen_Functions/',
        'Reweighted/',
        'Robust/',
        'DWGCS/',
    ]

    # Add paths
    import sys
    sys.path.append(path)
    for add_path in add_paths:
        sys.path.append(add_path)

    from phi import phi
    from Reweighted import Reweighted
    from Robust import Robust
    from DWGCS import DWGCS   

    Table_errors = np.zeros((10,7))
    Table_errorD_01 = np.zeros((10,10))
    Table_errorD_log = np.zeros((10,10))

    for i in range(4):

        dataset_normalize = Select_Dataset(i)
        d = dataset_normalize.shape[1]
        X = dataset_normalize[:,:d-1]

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
    
        RU_IW = np.zeros((100,1))
        error_IW = np.zeros((100,1))
        RU_Flat = np.zeros((100,1))
        error_Flat = np.zeros((100,1))
        RU_Rulsif = np.zeros((100,1))
        error_Rulsif = np.zeros((100,1))
        RU_Rob = np.zeros((100,1))
        error_Rob = np.zeros((100,1))
        RU_Kmm = np.zeros((100,1))
        error_Kmm = np.zeros((100,1))
        RU_best_Dwgcs_01 = np.zeros((100,1))
        error_best_Dwgcs_01 = np.zeros((100,1))
        RU_best_Dwgcs_log = np.zeros((100,1))
        error_best_Dwgcs_log = np.zeros((100,1))
        
        for rep in range(100):
            [Train_Set,Test_Set,n,t] = CovShiftGen.PCA(dataset_normalize)
            xtr = Train_Set[:,:d-1]
            ytr = Train_Set[:,d-1:].astype(int)
            xte = Test_Set[:,:d-1]
            yte = Test_Set[:,d-1:].astype(int)

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
            Rulsif = Reweighted.RuLSIF_fast(Rulsif,xtr,xte)
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
                RU_Dwgcs_01[l] = Dwgcs_01.RU
                error_Dwgcs_01[l] = Dwgcs_01.error

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
                RU_Dwgcs_log[l] = Dwgcs_log.RU
                error_Dwgcs_log[l] = Dwgcs_log.error

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