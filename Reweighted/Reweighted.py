import numpy as np
import cvxpy as cvx
import sklearn as sk
from phi import phi
from powerset import powerset

class Reweighted:

    def KMM(Mdl,xtr,xte):

        n = xtr.shape[0]
        t = xte.shape[0]
        x = np.concatenate((xtr,xte), axis=0)
        epsilon_ = 1-1/(np.sqrt(n))
        B = 1000

        K = sk.metrics.pairwise.rbf_kernel(x,x,1/(2*Mdl.sigma_**2))+(1e-15)*np.identity(n+t)
        kappa_=np.zeros((1,n))

        for i in range(n):
            for j in range(t):
                kappa_[0,i]=kappa_[0,i]+np.exp(-np.linalg.norm(xtr[i,:]-xte[j,:])**2/(2*Mdl.sigma_**2))

        # Define the variables of the opt. problem
        beta_ = cvx.Variable((n,1))
        alpha_ = cvx.Variable((t,1))
        # Define the objetive function
        objective = cvx.Minimize(cvx.quad_form(cvx.vstack([beta_/n, -alpha_/t]), K))
        # Define the constraints
        constraints = [ 
            beta_ >= np.zeros((n,1)),
            beta_ <= B * np.ones((n,1)),
            alpha_ == np.ones((t,1)),
            cvx.abs(cvx.sum(beta_)/n - 1) <= epsilon_,
        ]
        problem = cvx.Problem(objective,constraints)
        problem.solve(solver='MOSEK')
                      #,verbose=True)

        Mdl.beta_ = beta_.value
        Mdl.min_KMM = problem.value

        return Mdl
    
    def LREIW(Mdl,xtr, xte):
        n = xtr.shape[0]
        t = xte.shape[0]
    
        clf = sk.linear_model.LogisticRegression(penalty='l2', fit_intercept=False)
        clf.fit(np.vstack((xtr, xte)), np.concatenate((np.ones(n), -np.ones(t))))
    
        Mdl.beta_ = (n/t)*np.exp(xtr @ clf.coef_.T)
        Mdl.alpha_ = 1./((n/t)*np.exp(xte @ clf.coef_.T))
    
        return Mdl
    
    def RuSLIF(Mdl,xtr, xte):
        n = xtr.shape[0]
        t = xte.shape[0]
        x = np.concatenate((xtr,xte), axis=0)
        N = x.shape[0]
        lambda_ = 1e-5

        kappa_= np.zeros((t,1))

        K = sk.metrics.pairwise.rbf_kernel(x,x,1/(2*Mdl.sigma_**2))

        K1 = K[n:,n:]
        K2 = K[n:,:n]
        K = (0.5/t) * K1 @ K1 + (0.5/n) * K2 @ K2.T

        kappa_ = np.mean(K1, axis= 1)
        Mdl.K_ = K
        Mdl.kappa_ = kappa_

        # Define the variables of the opt. problem
        coef_ = cvx.Variable((t,1))
        # Define the objetive function
        objective = cvx.Minimize(0.5*cvx.quad_form(coef_, K) - cvx.matmul(kappa_.T, coef_) + 0.5*lambda_*cvx.quad_form(coef_, np.identity(t)))
        # Define the constraints
        constraints = []

        problem = cvx.Problem(objective,constraints)
        problem.solve(solver='MOSEK')
                      #,verbose=True)

        coef_ = np.maximum(coef_.value, 0)
        Mdl.min_RuSLIF = problem.value
        Mdl.beta_ = K2.T @ coef_

        return Mdl
    
    def parameters(Mdl,xtr,ytr):

        auxtau = []
        n = xtr.shape[0]

        for i in range(n):
            auxtau.append(Mdl.beta_[i] @ phi(Mdl,xtr[i, :],ytr[i]))
        Mdl.tau_ = np.ravel(np.mean(np.array(auxtau), axis=0)) 

        Mdl.lambda_ = Mdl.lambda0*np.std(np.array(auxtau),axis=0)/np.sqrt(n)

        return Mdl
    
    def learning(Mdl,xtr):
        n = xtr.shape[0]
        d = len(Mdl.tau_)

        if Mdl.loss == '0-1':

            v = np.zeros((2**Mdl.labels-1,1))

            set = powerset(Mdl.labels)
            # M = []
            M = np.empty((0,d))
            for i in range(n):
                for j in range(2**Mdl.labels-1):
                    M = np.vstack((M,np.sum(phi(Mdl,xtr[i,:],set[j]),axis=0)/set[j].shape[0]))
            # M = np.array(M)
            for j in range(2**Mdl.labels-1):
                v[j,0]=1/set[j].shape[0]
            v = np.tile(v,(1,n)) 
            # Define the variables of the opt. problem
            mu_ = cvx.Variable((d,1))
            # Define the objetive function
            objective = cvx.Minimize( -Mdl.tau_ @ mu_ \
                                    + cvx.sum( cvx.multiply(np.squeeze(Mdl.beta_),np.ones(n)+cvx.max(cvx.reshape(M @ mu_, (2**Mdl.labels-1, n)) - v, axis=0)))/n \
                                    + Mdl.lambda_ @ cvx.abs(mu_))
            problem = cvx.Problem(objective)
            problem.solve(solver='MOSEK')
        
        if Mdl.loss =='log':

            M = np.empty((0,d))
            for i in range(n):
                M = np.vstack( ( M,phi( Mdl,xtr[i,:],np.arange(1,Mdl.labels+1) ) ) )
            # Define the variables of the opt. problem
            mu_ = cvx.Variable((d,1))
            # Define the objetive function
            objective = cvx.Minimize( -Mdl.tau_ @ mu_ \
                                    + sum([Mdl.beta_[k]*cvx.log_sum_exp(M[2*k:2*k+2,:] @ mu_) for k in range(n)]) / n \
                                    + Mdl.lambda_ @ cvx.abs(mu_) )
            problem = cvx.Problem(objective)
            problem.solve(solver='MOSEK')

        Mdl.mu_ = mu_.value
        Mdl.RU = problem.value
        return Mdl   
    
    def prediction(Mdl,xte,yte):
        t = xte.shape[0]
        ye = np.zeros((t,1))

        if Mdl.deterministic == True:

            for i in range(t):
                ye[i] = np.argmax(phi(Mdl,xte[i,:],np.arange(1,Mdl.labels+1))@Mdl.mu_)+1
            Mdl.error = np.count_nonzero(yte != ye)/t

        if Mdl.deterministic == False:

            Mdl.h = np.zeros((Mdl.labels,t))

            if Mdl.loss == '0-1':
                set = powerset(Mdl.labels)
                varphi_mux = np.zeros(t)
                for i in range(t):
                    varphi_aux = np.zeros(2**Mdl.labels-1)
                    for j in range(2**Mdl.labels-1):
                        varphi_aux[j] = (np.sum( phi(Mdl,xte[i,:],set[j])@Mdl.mu_)-1)/set[j].shape[0]
                    varphi_mux[i]= max(varphi_aux)
                    c = np.sum(np.maximum( phi(Mdl,xte[i,:],np.arange(1,Mdl.labels+1)) @ Mdl.mu_-varphi_mux[i]*np.ones((Mdl.labels,1)),0))
                    if c == 0:
                        Mdl.h[:,i] = (1/Mdl.labels)*np.ones(Mdl.labels)
                    else:
                        Mdl.h[:,i] = np.squeeze(np.maximum( phi(Mdl,xte[i,:],np.arange(1,Mdl.labels+1)) @ Mdl.mu_-varphi_mux[i]*np.ones((Mdl.labels,1)),0)/c)
                    ye[i] = np.random.choice(np.arange(1,Mdl.labels+1), p=Mdl.h[:,i])
                Mdl.error = np.count_nonzero(yte != ye)/t
            
            if Mdl.loss == 'log':
                for i in range(t):
                    for j in range(Mdl.labels):
                        Mdl.h[j,i] = 1/sum(np.exp( phi(Mdl,xte[i,:],np.arange(1,Mdl.labels+1)) @ Mdl.mu_\
                                    -np.ones((Mdl.labels,1))* phi(Mdl,xte[i,:],np.array([j+1])) @ Mdl.mu_))
                    ye[i] = np.random.choice(np.arange(1,Mdl.labels+1), p=Mdl.h[:,i])
                Mdl.error = np.count_nonzero(yte != ye)/t
        return Mdl     