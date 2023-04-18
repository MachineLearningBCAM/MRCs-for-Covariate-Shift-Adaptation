import numpy as np
import cvxpy as cp
from Auxiliary_Functions.phi import phi

def DWGCS_parameters(Mdl,xtr,ytr,xte):

    auxtau = []
    n = xtr.shape[0]
    t = xte.shape[0]

    for i in range(n):
        auxtau.append(Mdl.beta[i] * phi(Mdl,xtr[i, :],ytr[i]))
    Mdl.tau = np.mean(np.array(auxtau), axis=0) 
    return Mdl
