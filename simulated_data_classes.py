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
from sklearn.neighbors import KNeighborsClassifier
dirname = os.getcwd()
from scipy.special import softmax

''' Constant Settings'''
I = 1 # Number of sequences
N = 200 # Number of instances
D = 0 # Dimension of the covariates
mr = 0.0 # Random missing rate
P = [1]  # of multinomial modalities
P += [0]  # of gaussian modalities
C_max = [100] * P[0] # Total number of counts - 1 (if 0 -> Categorical)

K = 2  # Latent space dimension
M = [25] * P[0] # Observation dimension
M = np.append(M, [20] * P[1]) # Observation dimension



def func(comb):
    
    K = comb[0]
    M[0] = comb[1]

    ''' Generate data'''
    data = data_gen(K, P, M, C_max, mr, D, static = True)
    X, y, mean = data.generate_data(N, I)
    X = [X[p][0] for p in range(sum(P))]
    
    
    ''' Train model'''
    model = multPCA(M[0].astype(int), K, D = D)
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
    
    '''Testing Class Seperability'''
    if True:

        Xc = []
        y = []
        z = []
        C = 3
        m = [[0,0], [1.5,1.5], [-1,-1]]
        s = [0.5, 0.5, 0.1]
        for c in range(C):
            data = data_gen(K, P, M, C_max, mr, D, static = True)
            data.mu0 = m[c]
            data.S0 = np.eye(2) * s[c]
            data.Q = np.eye(2) * s[c]
            X, _, mean = data.generate_data(N, I)
            z.append(np.array(data.z)[0])
            Xc.append(X[0][0])
            y += N * [c]
        Xc = np.vstack(Xc)
        y = np.array(y)
        z = np.vstack(z)
            
        ind = np.arange(C*N)
        np.random.shuffle(ind)
        ind_tr = ind[0:int(C*N*0.75)]
        ind_te = ind[int(C*N*0.75):]

        Xcte = Xc[ind_te]
        yte = y[ind_te]
        Xc = Xc[ind_tr]
        y = y[ind_tr]
        z = z[ind_tr]
        
        model = multPCA(M[0].astype(int), K, D = D)
        model.fit(Xc, y, epoch = 10000, step_comp_elbo = 1000)
        plt.plot(model.elbo_vec)
        plt.show()
        
        # Plot estimated embeddings
        colors = ['blue', 'red', 'green']
        markers = [ ".", "+", "x"]
        for c in range(C):
            plt.scatter(model.mu_z[np.where(y==c)[0],0], model.mu_z[np.where(y==c)[0],1], marker = markers[c], c = colors[c], alpha = 0.7)
        tmp = "output\\sim_classes" + ".pdf"
        filename = os.path.join(dirname, tmp)
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.savefig(filename)
        plt.show()
        
        # Plot orginal embeddings
        colors = ['blue', 'red', 'green']
        markers = [ ".", "+", "x"]
        for c in range(C):
            plt.scatter(z[np.where(y==c)[0],0], z[np.where(y==c)[0],1], marker = markers[c], c = colors[c], alpha = 0.7)
        tmp = "output\\sim_classes_org" + ".pdf"
        filename = os.path.join(dirname, tmp)
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.savefig(filename)
        plt.show()
        
        
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(Xc, y).score(Xcte, yte)
