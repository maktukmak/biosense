import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from scipy.special import softmax

from simulated_data_exps import data_gen

from scipy.stats import norm


def copula_transform(X):
    n = X.shape[0]
    eps = 1 / (4 * (n**0.25) * np.sqrt(np.pi * np.log(n))) 
    Xc = np.zeros(X.shape)
    for j in range(X.shape[1]):
        m = np.mean(X[:,j])
        s = np.std(X[:,j])
        
        quantiles, counts = np.unique(X[:,j], return_counts=True)
        cumprob = np.cumsum(counts).astype(np.double) / n
        Fj = np.array([cumprob[np.where(quantiles == X[:,j][i])[0][0]] for i in range(n)])
        Fj[Fj < eps] = eps
        Fj[Fj > 1 - eps] = 1 - eps
        hj = norm.ppf(Fj)
        fj = m + s * hj
        Xc[:,j] = fj
        
    return Xc

if __name__ == "__main__":

    # #############################################################################
    # Generate the data
    full_dim = True
    K = 5  # Latent space dimension
    P = [1]  # of modalities
    P += [0]
    N = 200
    D = 0
    M = [20] # Observation dimension
    C_max = np.random.randint(1000, 2000, P[0]) # Total number of counts - 1 (if 0 -> Categorical)
    
    # Prepare data
    Xm, X_te, prob, prob_te, W, _, _ = data_gen(K, P[0], N, M, C_max, full_dim)
    Xm = Xm[0].astype(np.float64)
    Xm_te = X_te[0].astype(np.float64)
    
    Ni = Xm[:, -1]
    X = np.append(Xm[:, :-1], (Ni - np.sum(Xm[:, :-1], axis = 1))[None].T, axis = 1)
    Ni = Xm_te[:, -1]
    X_te = np.append(Xm_te[:, :-1], (Ni - np.sum(Xm_te[:, :-1], axis = 1))[None].T, axis = 1) 
    
    
    Xl = np.log(X + 1e-6)
    Xc = copula_transform(Xl)


