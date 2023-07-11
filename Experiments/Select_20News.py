import numpy as np

def Select_20News(idx):
    if idx == 0:
        Train_Set = np.genfromtxt('comp_vs_sci_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('comp_vs_sci_TestSet', delimiter=',')
    if idx == 1:
        Train_Set = np.genfromtxt('comp_vs_talk_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('comp_vs_talk_TestSet', delimiter=',')
    if idx == 2:
        Train_Set = np.genfromtxt('rec_vs_sci_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('rec_vs_sci_TestSet', delimiter=',')
    if idx == 3:
        Train_Set = np.genfromtxt('rec_vs_talk_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('rec_vs_talk_TestSet', delimiter=',')
    if idx == 4:
        Train_Set = np.genfromtxt('sci_vs_talk_TrainSet.csv', delimiter=',') 
        Test_Set = np.genfromtxt('sci_vs_talk_TestSet', delimiter=',')  
    return Train_Set,Test_Set  