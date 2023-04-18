import numpy as np
import cvxpy as cvx

def DWKMM(Mdl,xtr,xte):

    n = xtr.shape[0]
    t = xte.shape[0]
    x = np.concatenate((xtr,xte), axis=0)
    epsilon = 1-1/(np.sqrt(n))
    B = 1000
    K=np.zeros((n+t,n+t))

    for i in range(n+t):
        K[i,i] = 0.5
        for j in range(i+1, n+t):
            K[i,j] = np.exp(-np.linalg.norm(x[i,:]-x[j,:])**2/(2*Mdl.sigma**2))
    K = K+np.transpose(K)
    
    # Define the variables of the opt. problem
    alpha = cvx.Variable((t,1))
    beta = cvx.Variable((n,1))
    # Define the objetive function
    objective = cvx.Minimize(cvx.quad_form(cvx.vstack([beta/n, alpha/t]), K))
    # Define the constraints
    constraints = [ 
        beta >= np.zeros((n,1)),
        beta <= (B/np.sqrt(Mdl.D)) * np.ones((n, 1)),
        alpha >= np.zeros((t,1)),
        alpha <= np.ones((t,1)),
        cvx.abs(cvx.sum(beta)/n - cvx.sum(alpha)/t) <= epsilon,
        cvx.norm(alpha - np.ones((t,1))) <= (1-1/np.sqrt(Mdl.D)) * np.sqrt(t)
    ]
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    Mdl.beta = beta.value
    Mdl.alpha = alpha.value
    Mdl.min_DWKMM = problem.value

    return Mdl