import numpy as np

class CovShiftGen:

    def PCA(dataset_normalize, n=None, t=None):
        datx = dataset_normalize[:, :-1]
        U, S, Vt = np.linalg.svd(datx)
        pc = Vt[0, :]
        dat_pca = datx @ pc
        m = np.median(dat_pca)

        Train_Set = []
        Test_Set = []

        for i in range(datx.shape[0]):
            if dat_pca[i] >= m:
                if np.random.rand() <= 0.7:
                    Train_Set.append(dataset_normalize[i, :])
                else:
                    Test_Set.append(dataset_normalize[i, :])
            if dat_pca[i] < m:
                if np.random.rand() <= 0.3:
                    Train_Set.append(dataset_normalize[i, :])
                else:
                    Test_Set.append(dataset_normalize[i, :])

        Train_Set = np.array(Train_Set)
        Test_Set = np.array(Test_Set)

        N = Train_Set.shape[0]
        T = Test_Set.shape[0]

        if n is not None and t is not None:
            n = min(n, N)
            t = min(t, T)
        else:
            if N >= 1000:
                n = 500
            else:
                n = round(N / 2)
            if T >= 1000:
                t = 500
            else:
                t = round(T / 2)

        idx_train = np.random.permutation(N)[:n]
        idx_test = np.random.permutation(T)[:t]

        Tr_Set = Train_Set[idx_train, :]
        Te_Set = Test_Set[idx_test, :]

        return Tr_Set, Te_Set, n, t  
    
    def Features(dataset_normalize, feature, n=None, t=None):
        Train_Set = []
        Test_Set = []
        med = np.median(dataset_normalize[:, feature])

        for i in range(dataset_normalize.shape[0]):
            if dataset_normalize[i, feature] <= med:
                if np.random.rand() <= 0.7:
                    Train_Set.append(dataset_normalize[i, :])
                else:
                    Test_Set.append(dataset_normalize[i, :])
            if dataset_normalize[i, feature] > med:
                if np.random.rand() <= 0.3:
                    Train_Set.append(dataset_normalize[i, :])
                else:
                    Test_Set.append(dataset_normalize[i, :])

        Train_Set = np.array(Train_Set)
        Test_Set = np.array(Test_Set)

        N = Train_Set.shape[0]
        T = Test_Set.shape[0]

        if n is not None and t is not None:
            n = min(n, N)
            t = min(t, T)
        else:
            if N >= 1000:
                n = 500
            else:
                n = round(N / 2)
            if T >= 1000:
                t = 500
            else:
                t = round(T / 2)

        idx_train = np.random.permutation(N)[:n]
        idx_test = np.random.permutation(T)[:t]

        Tr_Set = Train_Set[idx_train, :]
        Te_Set = Test_Set[idx_test, :]

        return Tr_Set, Te_Set, n, t
    
    def Features_BreastCancer(dataset, dataset_normalize, feature, n=None, t=None):
        Train_Set = []
        Test_Set = []

        for i in range(dataset.shape[0]):
            if dataset[i, feature] <= 5:
                if np.random.rand() <= 0.3:
                    Train_Set.append(dataset_normalize[i, :])
                else:
                    Test_Set.append(dataset_normalize[i, :])
            if dataset[i, feature] > 5:
                if np.random.rand() <= 0.7:
                    Train_Set.append(dataset_normalize[i, :])
                else:
                    Test_Set.append(dataset_normalize[i, :])

        Train_Set = np.array(Train_Set)
        Test_Set = np.array(Test_Set)

        N = Train_Set.shape[0]
        n = 100
        T = Test_Set.shape[0]
        t = 100

        idx_train = np.random.permutation(N)[:n]
        idx_test = np.random.permutation(T)[:t]

        Tr_Set = Train_Set[idx_train, :]
        Te_Set = Test_Set[idx_test, :]

        return Tr_Set, Te_Set, n, t