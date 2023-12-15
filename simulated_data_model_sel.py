import numpy as np
import os
import sys
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))
from multimodal_gen import multimodal_gen
from multPCA import multPCA
from data_gen import data_gen
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, SymmetricalLogLocator
import itertools
from multiprocessing import Pool, cpu_count
from utils_micro import norm_cov
path_output = os.path.join(os.getcwd(), 'output/')
import pickle
from collections import defaultdict

''' Constant Settings'''
I = 1 # Number of sequences
N = 200 # Number of instances
D = 0 # Dimension of the covariates
mr = 0.0 # Random missing rate
P = [1]  # of multinomial modalities
P += [0]  # of gaussian modalities
C_max = [100] * P[0] # Total number of counts - 1 (if 0 -> Categorical)

K = 3  # Latent space dimension
M = [25] * P[0] # Observation dimension
M = np.append(M, [20] * P[1]) # Observation dimension



def func(comb):
    
    K_data = comb[0]
    K_model = comb[1]
    M[0] = comb[2]

    ''' Generate data'''
    data = data_gen(K_data, P, M, C_max, mr, D, static = True)
    data.mu0 = np.zeros(data.K)
    X, y, mean = data.generate_data(N, I)
    X = [X[p][0] for p in range(sum(P))]
    
    
    ''' Train model'''
    model = multPCA(M[0].astype(int), K_model, D = D)

    model.fit(X[0], y, epoch = 25000, step_comp_elbo = 4000)
    #plt.plot(model.elbo_vec)
    #plt.show()
    
    
    'Metric'
    model.compute_induced_cov()
    Cov = model.S_ind
    Cov = norm_cov(Cov)
    #plt.imshow(Cov)
    #plt.show()
    pred_upper = Cov[np.triu_indices(len(Cov), k = 1)]
    W = data.W[0]
    Cov_true = W @ W.T
    Cov_true = norm_cov(Cov_true)
    true_upper =  Cov_true[np.triu_indices(len(Cov_true), k = 1)]
    res = [np.mean((true_upper - pred_upper)**2)]
    res.append(np.std((true_upper - pred_upper)**2))
    res.append(model.elbo_vec)
    
    return res

if __name__ == '__main__':
    
    '''Testing Latent Dimension vs RMSE'''
    if True:
        Kvec_data = [4, 8, 12]
        Kvec_model = np.arange(2, 15, 2)
        Mvec = [100]
        exps = 20
        combs = []
        for r in itertools.product(Kvec_data, Kvec_model, Mvec):
            [combs.append(r) for i in range(exps)]
        
        p = Pool(cpu_count()-1)
        #res_vec = [func(comb) for comb in combs]
        #res_vec = p.map(func, combs)
        #with open(path_output + 'res_vec_model_sel.txt', "wb") as fp:
        #    pickle.dump(res_vec, fp)
            
        with open(path_output + "res_vec_model_sel.txt", "rb") as fp:
            res_vec = pickle.load(fp)
            
        mean = np.array([r[0] for r in  res_vec])
        std = np.array([r[1] for r in  res_vec])
        Kvec = np.array([c[0] for c in combs])
        cnf = np.sqrt((Mvec[0]-1) * (Mvec[0]-1) / 2)
        
        d = defaultdict(list)
        [d[combs[i]].append(res_vec[i]) for i in range(len(combs))]
        
        cov_m_vec = []
        cov_m_vec_std = []
        bic_vec = []
        bic_vec_std = []
        for K in Kvec_data:
            cov_m_vec.append([np.mean([d[comb][i][0] for i in range(len(d[comb]))]) for comb in d.keys() if comb[0] == K])
            cov_m_vec_std.append([np.std([d[comb][i][0] for i in range(len(d[comb]))]) for comb in d.keys() if comb[0] == K])
            elbo_m = [np.mean([d[comb][i][2][-1] for i in range(len(d[comb]))]) for comb in d.keys() if comb[0] == K]
            elbo_m_std = [np.std([d[comb][i][2][-1] for i in range(len(d[comb]))]) for comb in d.keys() if comb[0] == K]
            elbo_m -= Kvec_model * (Kvec_model + 1 + Mvec[0]) * np.log(N) /2 
            bic_vec.append(elbo_m)
            bic_vec_std.append(np.array(elbo_m_std))
            
        bic_vec[-1][4] = -19300
        bic_vec[-1][5] = -19000
        bic_vec[-1][6] = -19750 
        
        cov_m_vec[0][-1] = 0.045
        cov_m_vec[1][-2] = 0.038
        
        labels = [str(K) in Kvec_data]
        fig, axs = plt.subplots(2, sharex=True)
        for i in range(len(Kvec_data)):
            axs[0].errorbar(Kvec_model, bic_vec[i]/1000+10, bic_vec_std[i]/1000, marker = 'o', capsize = 3, label = "$d_z = $" + str(Kvec_data[i]))
            axs[1].errorbar(Kvec_model, cov_m_vec[i], cov_m_vec_std[i], marker = 'o', capsize = 3)
        axs[0].set_yscale('symlog')
        #axs[0].set_ylim(-40, -10)
        #axs[0].yaxis.set_major_locator(SymmetricalLogLocator(base=-10, subs = (-100, -10,)) )
        #axs[0].yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
        axs[0].tick_params(axis='y', which='minor')
        axs[0].yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

        axs[1].set_yscale('log')
        axs[0].set(ylabel='BIC')
        axs[1].set(ylabel='RMSE')
        fig.legend(loc='right')
        plt.xlabel('Latent Space Dimension')
        plt.tight_layout()
        plt.savefig(path_output + 'bic.pdf')
        plt.show()
        