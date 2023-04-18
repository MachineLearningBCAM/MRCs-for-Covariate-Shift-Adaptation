import numpy as np
import cvxpy as cvx
from Auxiliary_Functions.phi import phi

def DWGCS_parameters(Mdl,xtr,ytr,xte):

    auxtau = []
    n = xtr.shape[0]
    t = xte.shape[0]

    for i in range(n):
        auxtau.append(Mdl.beta_[i] * phi(Mdl,xtr[i, :],ytr[i]))
    Mdl.tau_ = np.mean(np.array(auxtau), axis=0) 

    delta = 1e-6
    d = Mdl.tau_.shape[1]

    # Define the variables of the opt. problem
    lambda_ = cvx.Variable((1,d))
    p = cvx.Variable((t,Mdl.labels))
    # Define the objetive function
    objective = cvx.Minimize(cvx.sum(lambda_))
    # Define the constraints
    constraints = []
    for i in range(t):
        for j in range(Mdl.labels):
            aux = p[i,j] * Mdl.alpha_[i] * phi(Mdl,xte[i,:],np.array([j+1]))
            constraints.append(Mdl.tau_ - lambda_ + delta <= cvx.sum(aux))
            constraints.append(cvx.sum(aux) <= Mdl.tau_ + lambda_ - delta)
    constraints.append(lambda_ >= 0)
    constraints.append(cvx.sum(p,axis=1) == np.ones(t)/t)
    constraints.append(p >= 0)

    problem = cvx.Problem(objective, constraints)
    problem.solve()

    Mdl.lambda_ = np.maximum(lambda_.value,0)

    return Mdl
