import numpy as np

def Select_Dataset(idx):
    if idx == 0:
        dataset = np.genfromtxt('Datasets/Blood.csv', delimiter=',') 
        dataset_normalize = np.genfromtxt('Datasets/Blood_normalize.csv', delimiter=',') 
    if idx == 1:
        dataset = np.genfromtxt('Datasets/BreastCancer.csv', delimiter=',')
        dataset_normalize = np.genfromtxt('Datasets/BreastCancer_normalize.csv', delimiter=',') 
    if idx == 2:
        dataset = np.genfromtxt('Datasets/Haberman_normalize.csv', delimiter=',') 
        dataset_normalize = np.genfromtxt('Datasets/Haberman_normalize.csv', delimiter=',') 
    if idx == 3:
        dataset_normalize = np.genfromtxt('Datasets/Ringnorm.csv', delimiter=',')
        dataset_normalize = np.genfromtxt('Datasets/Ringnorm_normalize.csv', delimiter=',') 
    return dataset,dataset_normalize  