import numpy as np

def Select_20News(idx):
    if idx == 0:
        X_Train = np.genfromtxt('Datasets_20News/comp_vs_sci_X_Train.csv', delimiter=',')
        Y_Train = np.genfromtxt('Datasets_20News/comp_vs_sci_Y_Train.csv', delimiter=',')       
        X_Test = np.genfromtxt('Datasets_20News/comp_vs_sci_X_Test.csv', delimiter=',')
        Y_Test = np.genfromtxt('Datasets_20News/comp_vs_sci_Y_Test.csv', delimiter=',')
    if idx == 1:
        X_Train = np.genfromtxt('Datasets_20News/comp_vs_talk_X_Train.csv', delimiter=',')
        Y_Train = np.genfromtxt('Datasets_20News/comp_vs_talk_Y_Train.csv', delimiter=',')       
        X_Test = np.genfromtxt('Datasets_20News/comp_vs_talk_X_Test.csv', delimiter=',')
        Y_Test = np.genfromtxt('Datasets_20News/comp_vs_talk_Y_Test.csv', delimiter=',')
    if idx == 2:
        X_Train = np.genfromtxt('Datasets_20News/rec_vs_sci_X_Train.csv', delimiter=',')
        Y_Train = np.genfromtxt('Datasets_20News/rec_vs_sci_Y_Train.csv', delimiter=',')       
        X_Test = np.genfromtxt('Datasets_20News/rec_vs_sci_X_Test.csv', delimiter=',')
        Y_Test = np.genfromtxt('Datasets_20News/rec_vs_sci_Y_Test.csv', delimiter=',')
    if idx == 3:
        X_Train = np.genfromtxt('Datasets_20News/rec_vs_talk_X_Train.csv', delimiter=',')
        Y_Train = np.genfromtxt('Datasets_20News/rec_vs_talk_Y_Train.csv', delimiter=',')       
        X_Test = np.genfromtxt('Datasets_20News/rec_vs_talk_X_Test.csv', delimiter=',')
        Y_Test = np.genfromtxt('Datasets_20News/rec_vs_talk_Y_Test.csv', delimiter=',')
    if idx == 4:
        X_Train = np.genfromtxt('Datasets_20News/sci_vs_talk_X_Train.csv', delimiter=',')
        Y_Train = np.genfromtxt('Datasets_20News/sci_vs_talk_Y_Train.csv', delimiter=',')       
        X_Test = np.genfromtxt('Datasets_20News/sci_vs_talk_X_Test.csv', delimiter=',')
        Y_Test = np.genfromtxt('Datasets_20News/sci_vs_talk_Y_Test.csv', delimiter=',') 
    return X_Train, Y_Train, X_Test, Y_Test  