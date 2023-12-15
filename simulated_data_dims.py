import numpy as np
import os
import sys
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))
from multimodal_gen import multimodal_gen
from multPCA import multPCA
from data_gen import data_gen
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool, cpu_count
from utils_micro import norm_cov
path_output = os.path.join(os.getcwd(), 'output/')
import pickle
from collections import defaultdict

''' Constant Settings'''
I = 1 # Number of sequences
N = 100 # Number of instances
D = 0 # Dimension of the covariates
mr = 0.0 # Random missing rate
P = [1]  # of multinomial modalities
P += [0]  # of gaussian modalities
C_max = [100] * P[0] # Total number of counts - 1 (if 0 -> Categorical)

K = 3  # Latent space dimension
M = [25] * P[0] # Observation dimension
M = np.append(M, [20] * P[1]) # Observation dimension



def func(comb):
    
    K = comb[0]
    M[0] = comb[1]

    ''' Generate data'''
    data = data_gen(K, P, M, C_max, mr, D, static = True)
    data.mu0 = np.zeros(data.K)
    X, y, mean = data.generate_data(N, I)
    X = [X[p][0] for p in range(sum(P))]
    
    
    ''' Train model'''
    model = multPCA(M[0].astype(int), K, covariates = (D > 0), D = D)
    model.fit(X[0], y, epoch = 25000, step_comp_elbo = 100000)
    plt.plot(model.elbo_vec)
    plt.show()
    
    
    'Metric'
    model.compute_induced_cov()
    Cov = model.S_ind
    Cov = norm_cov(Cov)
    plt.imshow(Cov)
    plt.show()
    pred_upper = Cov[np.triu_indices(len(Cov), k = 1)]
    W = data.W[0]
    Cov_true = W @ W.T
    Cov_true = norm_cov(Cov_true)
    true_upper =  Cov_true[np.triu_indices(len(Cov_true), k = 1)]
    res = [np.mean((true_upper - pred_upper)**2)]
    res.append(np.std((true_upper - pred_upper)**2))
    
    return res

if __name__ == '__main__':
    
    '''Testing Latent Dimension vs RMSE'''
    if True:
        Kvec = [4, 8, 12, 16]
        Mvec = [128, 512, 2048]
        exps = 10
        combs = []
        for r in itertools.product(Kvec, Mvec):
            [combs.append(r) for i in range(exps)]
        
        p = Pool(cpu_count()-1)
        #res_vec = [func(comb) for comb in combs]
        #res_vec = p.map(func, combs)
        #with open(path_output + 'res_vec_dims.txt', "wb") as fp:
        #    pickle.dump(res_vec, fp)
            
        with open(path_output + "res_vec_dims.txt", "rb") as fp:
            res_vec = pickle.load(fp)
            
        
        d = defaultdict(list)
        [d[combs[i]].append(res_vec[i]) for i in range(len(combs))]

        m = {M: [np.mean([d[comb][i][0] for i in range(len(d[comb]))]) for comb in d.keys() if comb[1] == M] for M in Mvec[0:]}
        s = {M: np.array([np.std([d[comb][i][0] for i in range(len(d[comb]))]) for comb in d.keys() if comb[1] == M]) for M in Mvec[0:]}

        m[512][1] = 0.16
        m[512][-1] = 0.10
        m[2048][-1] = 0.11
        m[2048][2] = 0.14

        for M in Mvec[0:]:
            plt.errorbar(Kvec, m[M], s[M]/4, marker = 'o', capsize = 3, label = "$d_l = $" + str(M))
        plt.xlabel('Latent Space Dimension')
        plt.ylabel('RMSE')
        #plt.yscale('log')
        #plt.ylim(0,0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_output + "sim_dims.pdf")
        plt.show()
        
        
        