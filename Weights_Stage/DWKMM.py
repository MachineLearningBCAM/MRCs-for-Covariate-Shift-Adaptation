import numpy as np
import cvxpy as cvx

def DWKMM(Mdl,xtr,xte):

    n = xtr.shape[0]
    t = xte.shape[0]
    x = np.concatenate((xtr,xte), axis=0)
    epsilon_ = 1-1/(np.sqrt(n))
    B = 1000
    K=np.zeros((n+t,n+t))

    for i in range(n+t):
        K[i,i] = 0.5
        for j in range(i+1, n+t):
            K[i,j] = np.exp(-np.linalg.norm(x[i,:]-x[j,:])**2/(2*Mdl.sigma_**2))
    K = K+np.transpose(K)
    
    # Define the variables of the opt. problem
    alpha_ = cvx.Variable(t)
    beta_ = cvx.Variable(n)
    # Define the objetive function
    objective = cvx.Minimize(cvx.quad_form(cvx.hstack([beta_/n, alpha_/t]), K))
    # Define the constraints
    constraints = [ 
        beta_ >= np.zeros(n),
        beta_ <= (B/np.sqrt(Mdl.D)) * np.ones(n),
        alpha_ >= np.zeros(t),
        alpha_ <= np.ones(t),
        cvx.abs(cvx.sum(beta_)/n - cvx.sum(alpha_)/t) <= epsilon_,
        cvx.norm(alpha_ - np.ones(t)) <= (1-1/np.sqrt(Mdl.D)) * np.sqrt(t)
    ]

    problem = cvx.Problem(objective, constraints)
    problem.solve()

    Mdl.beta_ = beta_.value
    Mdl.alpha_ = alpha_.value
    Mdl.min_DWKMM = problem.value

    return Mdl