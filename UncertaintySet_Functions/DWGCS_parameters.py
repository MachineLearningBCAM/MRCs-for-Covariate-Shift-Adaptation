import numpy as np
import cvxpy as cp

def DWGCS_parameter(Mdl,xtr,ytr,xte):

    auxtau = []
    n = xtr.shape[0]
    t = xts.shape[0]

    for i in range(n):
        auxtau.append(Mdl.beta[i] * phi(Mdl,xtr[i, :],ytr[i]))
    Mdl.tau = np.sum(np.array(auxtau))/n
