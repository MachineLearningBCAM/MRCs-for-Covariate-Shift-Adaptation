import numpy as np
import cvxpy as cvx
import sklearn as sk
from Auxiliary_Functions.phi import phi
from Auxiliary_Functions.powerset import powerset

class MRC:

    def parameters(Mdl,xtr,ytr):

        auxtau = []
        n = xtr.shape[0]

        for i in range(n):
            auxtau.append( phi(Mdl,xtr[i, :],ytr[i]))
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
                                    + cvx.sum(np.ones(n)+cvx.max(cvx.reshape(M @ mu_, (2**Mdl.labels-1, n)) - v, axis=0))/n \
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
                                    + sum([cvx.log_sum_exp(M[3*k:3*k+3,:] @ mu_) for k in range(n)]) / n \
                                    + Mdl.lambda_ * cvx.abs(mu_) )
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
                
                    

        
        