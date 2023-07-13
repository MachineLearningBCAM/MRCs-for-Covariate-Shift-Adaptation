import numpy as np

def Select_20News(idx):
    if idx == 0:
        Train_Set = np.genfromtxt('Datasets/comp_vs_sci_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('Datasets/comp_vs_sci_TestSet.csv', delimiter=',')
    if idx == 1:
        Train_Set = np.genfromtxt('Datasets/comp_vs_talk_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('Datasets/comp_vs_talk_TestSet.csv', delimiter=',')
    if idx == 2:
        Train_Set = np.genfromtxt('Datasets/rec_vs_sci_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('Datasets/rec_vs_sci_TestSet.csv', delimiter=',')
    if idx == 3:
        Train_Set = np.genfromtxt('Datasets/rec_vs_talk_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('Datasets/rec_vs_talk_TestSet.csv', delimiter=',')
    if idx == 4:
        Train_Set = np.genfromtxt('Datasets/sci_vs_talk_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('Datasets/sci_vs_talk_TestSet.csv', delimiter=',')  
    return Train_Set,Test_Set  