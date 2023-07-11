import numpy as np
import sklearn as sk
from sklearn.decomposition import PCA

class CGS:

    def  Features(dataset,feature,p1,p2):

        Train_Set = []
        Test_Set  = []

        med = np.median(dataset[:,feature-1]) 

        for i in range(dataset.shape[0]):
            if dataset[i,feature] <= med:
                if np.random.random_sample() <= p1:
                    Train_Set.append(dataset[i,:])
                else:
                    Test_Set.append(dataset[i,:])  
            else:
                if np.random.random_sample() <= p2:
                    Train_Set.append(dataset[i,:])
                else:
                    Test_Set.append(dataset[i,:])
    
        N = Train_Set.shape[0]
        T = Test_Set.shape[0]

        if N>= 1000:
            n = 500
        else:
            n = np.round(N/2)
        if T>= 1000:
            t = 500
        else:
            t = np.round(T/2)

        np.random.shuffle(Train_Set)
        np.random.shuffle(Test_Set)

        Train_Set = Train_Set[:n,:]
        Test_Set  = Test_Set[:t,:]

        return Train_Set,Test_Set,n,t
    
    def  PCA(dataset,p1,p2,n,t):

        Train_Set = []
        Test_Set  = []

        datx = dataset[:, :-1]
        pca = sk.decomposition.PCA
        FV = pca.fit_transform(datx)
        pc = FV[:, 0]
        dataset_pca = datx @ pc
        med = np.median(dataset_pca) 

        for i in range(dataset.shape[0]):
            if dataset_pca[i] >= med:
                if np.random.random_sample() <= p1:
                    Train_Set.append(dataset[i,:])
                else:
                    Test_Set.append(dataset[i,:])  
            else:
                if np.random.random_sample() <= p2:
                    Train_Set.append(dataset[i,:])
                else:
                    Test_Set.append(dataset[i,:])
    
        N = Train_Set.shape[0]
        T = Test_Set.shape[0]

        try:
            n
            # Variable 'n' exists
            print("Variable 'n' exists.")
        except NameError:
            # Variable 'n' does not exist
            print("Variable 'n' does not exist.")

        if N>= 1000:
            n = 500
        else:
            n = np.round(N/2)
        if T>= 1000:
            t = 500
        else:
            t = np.round(T/2)

        np.random.shuffle(Train_Set)
        np.random.shuffle(Test_Set)

        Train_Set = Train_Set[:n,:]
        Test_Set  = Test_Set[:t,:]

        return Train_Set,Test_Set,n,t