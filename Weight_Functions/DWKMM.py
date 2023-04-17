import numpy as np
import cvxpy as cp

def DWKMM(Mdl,xtr,xte):

    n = xtr.shape[0]
    t = xte.shape[0]
    x = np.concatenate((xtr,xte), axis=0)
    epsilon = 1-1/(np.sqrt(n))
    K=np.zeros((n+t,n+t))

    for i in range(n+t):
        K[i,i] = 0.5
        for j in range(i+1, n+t):
            K[i,j] = np.exp(-np.linalg.norm(x[i,:]-x[j,:])**2/(2*Mdl.sigma**2))
    K = K+np.transpose(K)
    
    # Define the variables of the opt. problem
    alpha = cp.Variable((t,1))
    beta = cp.Variable((n,1))
    # Define the objetive function
    objective = cp.Minimize(cp.quad_form(cp.vstack([beta/n, alpha/t]), K))
    # Define the constraints
    constraints = [ 
        beta >= np.zeros((n,1)),
        beta <= (Mdl.B/np.sqrt(Mdl.D)) * np.ones((n, 1)),
        alpha >= np.zeros((t,1)),
        alpha <= np.ones((t,1)),
        cp.abs(cp.sum(beta)/n - cp.sum(alpha)/t) <= epsilon,
        cp.norm(alpha - np.ones((t,1))) <= (1-1/np.sqrt(Mdl.D)) * np.sqrt(t)
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    Mdl.beta = beta.value
    Mdl.alpha = alpha.value
    Mdl.min_DWKMM = problem.value

    return Mdl