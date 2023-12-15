
import numpy as np
from scipy.linalg import norm
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import invwishart, invgamma
import matplotlib.pyplot as plt
from scipy.special import loggamma, multigammaln, gamma, digamma, softmax, logsumexp
import seaborn as sns
from scipy.optimize import minimize
sns.set_theme()
import matplotlib.pyplot as plt
import time

import os
import sys
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))
from multimodal_gen import multimodal_gen



# Data generation
def data_gen(K, P, N, M, C_max, full_dim = False, covariates = False, D = 0):
    
    U = 0
    U_te = 0
    if covariates:
        U = np.random.multivariate_normal(np.zeros(D), np.eye(D), size = N)
        V = np.random.multivariate_normal(np.zeros(D), np.eye(D), size = K)
        mu = U @ V.T
        z = np.array([np.random.multivariate_normal(mu[i], np.eye(K)) for i in range(N)])
    else:
        z = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = N)
    
    
    if full_dim:
        W = [np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M[p]+1) for p in range(P)]
    else:
        W = [np.random.multivariate_normal(np.zeros(K), np.eye(K), size = M[p]) for p in range(P)]
    C = [np.random.poisson(C_max[p],N) + 1 for p in range(P)]
    
    Th = [z @ W[p].T for p in range(P)]
    Th_ext = [np.append(Th[p], np.zeros((Th[p].shape[0],1)), axis = 1) for p in range(P)]
    
    if full_dim:
        prob = [softmax(Th[p], axis = 1) for p in range(P)]
    else:
        prob = [softmax(Th_ext[p], axis = 1) for p in range(P)]
    
    X = [np.array([np.random.multinomial(C[p][i], prob[p][i])[:-1] for i in range(N)]) for p in range(P)]
    X = [np.append(X[p], C[p][None].T, axis = 1) for p in range(P)]   #Last column is total number of counts
    
    X_te = [X[p][int(N*0.8):,:] for p in range(P)]
    X = [X[p][:int(N*0.8),:] for p in range(P)]
    if covariates:
        U_te = U[int(N*0.8):,:]
        U = U[:int(N*0.8),:]
        
    prob_te = [prob[p][int(N*0.8):,:] for p in range(P)]
    prob = [prob[p][:int(N*0.8),:] for p in range(P)]
    
    return X, X_te, prob, prob_te, W, U, U_te

if __name__ == "__main__":
           

    if True: # Estimate precision matrix
        
        full_dim = True
        
        K = 3  # Latent space dimension
        P = [1]  # of modalities
        P += [0]
        N = 200
        D = 0
        M = np.random.randint(20, 30, P[0]) # Observation dimension
        C_max = np.random.randint(1000, 2000, P[0]) # Total number of counts - 1 (if 0 -> Categorical)
        
        X, X_te, prob, prob_te, W = data_gen(K, P[0], N, M, C_max,full_dim)
        
        model = multimodal_gen(P, M, K = K, step_comp_elbo = 50, covariates = False, D = D, full_dim = full_dim)
        model.fit(X, epoch = 500)
        
        plt.plot(model.elbo_vec)
        
        
        prec = [W[i] @ W[i].T for i in range(P[0])]
        prec_p = [model.comps[i].W @ model.comps[i].S_prior @ model.comps[i].W.T for i in range(P[0])]
        
        print('MSE:', np.mean([np.mean((prec[i] - prec_p[i]) ** 2)  for i in range(P[0])]))
        
    
    if False: # Estimate the last test modality given the other modalities
        
        K = 3  # Latent space dimension
        P = 5  # of modalities
        N = 200
        M = np.random.randint(20, 30, P) # Observation dimension
        C_max = np.random.randint(1000, 2000, P) # Total number of counts - 1 (if 0 -> Categorical)
        
        X, X_te, prob = data_gen(K, P, N, M, C_max)
        
        # Fit
        model = multimodal_gen(P, M, K)
        model.fit(X, epoch = 200)
    
        
        Nte = N - int(N*0.8)
        print('Results')
        for p in range(P):
            mask = np.zeros((Nte, len(X)))
            mask[:,:p] = 1
            prob_est = model.predict([X_te[p][:Nte,:] * mask[:,p,None] for p in range(P)] )
            d_hel = np.mean(norm(np.sqrt(prob_est[-1]) - np.sqrt(prob[-1][int(N*0.8):,:]), axis = 1) / np.sqrt(2))
            print(d_hel)
            
    
            
    if False: # Acc vs total number of counts
        
        K = 3  # Latent space dimension
        P = 2  # of modalities
        N = 200
        M = [25]*P  # Observation dimension
        
        exp = 5
        
        d_hel_vec = []
        rng = np.arange(0, 100, 10)
        for C_max in rng:
            
            d_hel = []
            for e in range(exp):
            
                X, X_te, prob, prob_te = data_gen(K, P, N, M, [C_max]*P)
                
                model = multimodal_gen(P, M, K, step_comp_elbo = 500)
                model.fit(X, epoch = 200)
                
                mask = np.zeros((len(X_te[0]), len(X)))
                mask[:,0] = 1
                prob_est = model.predict([X_te[p] * mask[:,p,None] for p in range(P)])
                
                d_hel.append(np.mean(norm(np.sqrt(prob_est[-1]) - np.sqrt(prob_te[-1]), axis = 1) / np.sqrt(2)))
            d_hel_vec.append([np.mean(d_hel), np.std(d_hel)])
            
        d_hel_vec = np.array(d_hel_vec)
        
        plt.plot(rng, d_hel_vec[:,0])
        plt.fill_between(rng,
                         d_hel_vec[:,0] - d_hel_vec[:,1],
                         d_hel_vec[:,0] + d_hel_vec[:,1],
                         color='blue', alpha=0.1)
        plt.xlabel('Total number of counts per sample')
        plt.ylabel('Hellinger distance')
        plt.show()
        
        
    if False: # Acc vs N
        
        K = 3  # Latent space dimension
        P = 2  # of modalities
        #N = 200
        M = [25]*P  # Observation dimension
        C_max = 20
        
        exp = 20
        
        d_hel_vec = []
        rng = np.arange(10, 100, 20)
        for N in rng:
            
            d_hel = []
            for e in range(exp):
            
                X, X_te, prob, prob_te = data_gen(K, P, N, M, [C_max]*P)
                
                model = multimodal_gen(P, M, K, step_comp_elbo = 500)
                model.fit(X, epoch = 200)
                
                mask = np.zeros((len(X_te[0]), len(X)))
                mask[:,0] = 1
                prob_est = model.predict([X_te[p] * mask[:,p,None] for p in range(P)])
                
                d_hel.append(np.mean(norm(np.sqrt(prob_est[-1]) - np.sqrt(prob_te[-1]), axis = 1) / np.sqrt(2)))
            d_hel_vec.append([np.mean(d_hel), np.std(d_hel)])
            
        d_hel_vec = np.array(d_hel_vec)
        
        plt.plot(rng, d_hel_vec[:,0])
        plt.fill_between(rng,
                         d_hel_vec[:,0] - d_hel_vec[:,1],
                         d_hel_vec[:,0] + d_hel_vec[:,1],
                         color='blue', alpha=0.1)
        plt.xlabel('Total number of samples')
        plt.ylabel('Hellinger distance')
        plt.show()
        
    if False: # Acc vs P
        
        K = 3  # Latent space dimension
        #P = 2  # of modalities
        N = 20
        
        C_max = 20
        
        exp = 10
        
        d_hel_vec = []
        rng = np.arange(1, 10, 1)
        for P in rng:
            M = [25]*P  # Observation dimension
            
            d_hel = []
            for e in range(exp):
            
                X, X_te, prob, prob_te = data_gen(K, P, N, M, [C_max]*P)
                
                model = multimodal_gen(P, M, K, step_comp_elbo = 500)
                model.fit(X, epoch = 200)
                
                mask = np.zeros((len(X_te[0]), len(X)))
                mask[:,:-1] = 1
                prob_est = model.predict([X_te[p] * mask[:,p,None] for p in range(P)])
                
                d_hel.append(np.mean(norm(np.sqrt(prob_est[-1]) - np.sqrt(prob_te[-1]), axis = 1) / np.sqrt(2)))
            d_hel_vec.append([np.mean(d_hel), np.std(d_hel)])
            
        d_hel_vec = np.array(d_hel_vec)
        
        plt.plot(rng, d_hel_vec[:,0])
        plt.fill_between(rng,
                         d_hel_vec[:,0] - d_hel_vec[:,1],
                         d_hel_vec[:,0] + d_hel_vec[:,1],
                         color='blue', alpha=0.1)
        plt.xlabel('Total number of modalities')
        plt.ylabel('Hellinger distance')
        plt.show()
        
        
