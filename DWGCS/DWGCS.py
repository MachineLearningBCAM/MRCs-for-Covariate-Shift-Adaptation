import numpy as np
import cvxpy as cvx
from Auxiliary_Functions.phi import phi
from Auxiliary_Functions.powerset import powerset

class DWGCS:

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

    def parameters(Mdl,xtr,ytr,xte):

        auxtau = []
        n = xtr.shape[0]
        t = xte.shape[0]

        for i in range(n):
            auxtau.append(Mdl.beta_[i] * phi(Mdl,xtr[i, :],ytr[i]))
        Mdl.tau_ = np.ravel(np.mean(np.array(auxtau), axis=0)) 

        delta = 1e-6
        d = len(Mdl.tau_)

        # Define the variables of the opt. problem
        lambda_ = cvx.Variable(d)
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
    
    def learning(Mdl,xte):
        t = xte.shape[0]
        d = len(Mdl.tau_)

        if Mdl.loss == '0-1':

            v = np.zeros((2**Mdl.labels-1,1))

            set = powerset(Mdl.labels)
            # M = []
            M = np.empty((0,d))
            for i in range(t):
                for j in range(2**Mdl.labels-1):
                    # M.append(np.sum(Mdl.alpha_[i]*phi(Mdl,xte[i,:],set[j]),axis=0)/set[j].shape[0])
                    M= np.vstack((M,np.sum(Mdl.alpha_[i]*phi(Mdl,xte[i,:],set[j]),axis=0)/set[j].shape[0]))
            # M= np.array(M)
            for j in range(2**Mdl.labels-1):
                v[j,0]=1/set[j].shape[0]
            v = np.tile(v,(1,t)) 
            # Define the variables of the opt. problem
            mu_ = cvx.Variable((d,1))
            # Define the objetive function
            objective = cvx.Minimize( -Mdl.tau_ @ mu_ \
                                    + cvx.sum(np.ones((1,t))+cvx.max(cvx.reshape(M @ mu_, (2**Mdl.labels-1, t)) - v))/t \
                                    + Mdl.lambda_ @ cvx.abs(mu_))
            problem = cvx.Problem(objective)
            problem.solve()
        
        if Mdl.loss =='log':

            M = np.empty((0,d))
            for i in range(t):
                M = np.vstack( ( M,Mdl.alpha_[i]*phi( Mdl,xte[i,:],np.arange(1,Mdl.labels+1) ) ) )
            # Define the variables of the opt. problem
            mu_ = cvx.Variable((d,1))
            # Define the objetive function
            objective = cvx.Minimize( -Mdl.tau_ @ mu_ \
                                    + sum([cvx.log_sum_exp(M[3*k:3*k+3,:] @ mu_) for k in range(t)]) / t \
                                    + Mdl.lambda_ * cvx.abs(mu_) )
            problem = cvx.Problem(objective)
            problem.solve()

        Mdl.mu_ = mu_.value
        Mdl.RU = problem.value
        return Mdl   
    
    def prediction(Mdl,xte,yte):
        t = xte.shape[0]
        error = 0
        ye = np.zeros((t,1))

        if Mdl.deterministic == True:

            for i in range(t):
                ye[i] = np.argmax(phi(Mdl,xte[i,:],np.arange(1,Mdl.labels+1))*Mdl.mu_)+1

        if Mdl.deterministic == False:

            Mdl.h = np.zeros((Mdl.labels,t))

            if Mdl.loss == '0-1':
                set = powerset(Mdl.labels)
                varphi_mux = np.array(t)
                for i in range(t):
                    varphi_aux = np.zeros(2**Mdl.labels-1)
                    for j in range(2**Mdl.labels-1):
                        varphi_aux[j] = (np.sum(phi(Mdl,xte[i,:],set[j])*Mdl.mu_)-1)/set[j].shape[0]
                    varphi_mux[i]= max(varphi_aux)
                    c = np.sum(np.maximum(phi(Mdl,xte[i,:],np.arange(1,Mdl.labels+1))*Mdl.mu_-np.ones((Mdl.labels,1)*varphi_mux[i]),0))
                    if c == 0:
                        Mdl.h[:,i] = (1/Mdl.labels)*np.ones((Mdl.labels,1))
                    else:
                        Mdl.h[:,i] = np.maximum(phi(Mdl,xte[i,:],np.arange(1,Mdl.labels+1))*Mdl.mu_-np.ones((Mdl.labels,1))*varphi_mux[i],0)/c
                    ye[i] = np.ranfom.choice(np.arange(1,Mdl.labels+1), p=Mdl.h[:,i])+1
                error = np.count_nonzero(yte != ye)/t
            
            if Mdl.loss == 'log':
                for i in range(t):
                    for j in range(Mdl.labels):
                        Mdl.h[j,i] = 1/sum(np.exp(phi(Mdl,xte[i,:],np.arange(1,Mdl.labels+1))*Mdl.mu_\
                                    -np.ones((Mdl.labels,1))*phi(Mdl,xte[i,:],np.array([j+1]))*Mdl.mu_))
                    ye[i] = np.ranfom.choice(np.arange(1,Mdl.labels+1), p=Mdl.h[:,i])+1
                error = np.count_nonzero(yte != ye)/t
        return Mdl
                
                    

        
        