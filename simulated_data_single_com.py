import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from scipy.special import softmax

from simulated_data_exps import data_gen
import os
import sys
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))
dirname = os.getcwd()

from multimodal_gen import multimodal_gen
from multPCA import multPCA
from copula import copula_transform

import pickle
import seaborn as sns

# #############################################################################
# Generate the data
data = 'gen_low_rank'  #'gen_low_rank', 'gen_sparse_prec', 'bio' 

if data == 'gen_low_rank':

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
    
    # Data for numerical model
    Ni = Xm[:, -1]
    X = np.append(Xm[:, :-1], (Ni - np.sum(Xm[:, :-1], axis = 1))[None].T, axis = 1)
    Ni = Xm_te[:, -1]
    X_te = np.append(Xm_te[:, :-1], (Ni - np.sum(Xm_te[:, :-1], axis = 1))[None].T, axis = 1)   
    m = X.mean(axis=0)
    s = X.std(axis=0)
    X -= m
    X /= s
    X_te -= m
    X_te /= s
    
    
    # Ground truth covariance
    A = 0.5*(np.eye(M[0]+1) - (1/(M[0]+2))*np.ones((M[0]+1, M[0]+1)))
    cov = W[0] @ W[0].T + np.linalg.inv(A)
    prec = linalg.inv(cov)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    
    
    
elif data == 'gen_sparse_prec':

    N = 200
    n_features = 20
    
    prng = np.random.RandomState(1)
    prec = make_sparse_spd_matrix(
        n_features, alpha=0.98, smallest_coef=0.4, largest_coef=0.7, random_state=prng
    )
    cov = linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=N)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    
    Xm = np.around(np.exp(X))
    Xm[:,-1] = Xm.sum(axis = 1)
    M = [Xm.shape[1]-1]
    P = [1] + [0]
    K = 3
    D = 0

def norm_cov(cov, prec):
    d = np.sqrt(np.diag(cov))
    cov = cov / d
    cov /= d[:, np.newaxis]
    prec = prec * d
    prec *= d[:, np.newaxis]
    
    d = np.sqrt(np.diag(prec))
    prec = prec / d
    prec /= d[:, np.newaxis]
    
    return cov, prec


if False: # Test Elbo wrt rank

    elbo_vec_exp = []
    rmse_vec_exp = []
    for e in range(10):
    
        K_vec = np.arange(2,12)
        elbo_vec = []
        cov_vec = []
        for k in K_vec:
            model = multimodal_gen(P = P, M = M, K = k, step_comp_elbo = 200, covariates = None, D = 0, full_dim =  True)
            model.fit([Xm], epoch = 2000)
            elbo_vec.append(model.compute_test_elbo([Xm_te]))
            m_covo = model.S_ind
            m_cov,_ = norm_cov(m_covo, m_covo)
            cov_vec.append(m_cov)
        
        rmse_vec = np.sqrt(np.mean((cov_vec - cov)**2, axis = (1,2)))
        
        elbo_vec_exp.append(elbo_vec)
        rmse_vec_exp.append(rmse_vec)
    
    
    plt.plot(K_vec, np.mean(elbo_vec_exp, axis = 0))
    plt.fill_between(K_vec, 
                     np.mean(elbo_vec_exp, axis = 0) - np.std(elbo_vec_exp, axis = 0),
                     np.mean(elbo_vec_exp, axis = 0) + np.std(elbo_vec_exp, axis = 0),
                     color='black', alpha=0.2)
    plt.xlabel('Latent space dimension')
    plt.ylabel('Test ELBO')
    tmp = "output\\test_elbo_vs_rank.pdf"
    filename = os.path.join(dirname, tmp)
    plt.tight_layout()
    plt.savefig(filename, quality = 95)
    plt.show()
    
    
    plt.plot(K_vec, np.mean(rmse_vec_exp, axis = 0))
    plt.fill_between(K_vec, 
                     np.mean(rmse_vec_exp, axis = 0) - np.std(rmse_vec_exp, axis = 0),
                     np.mean(rmse_vec_exp, axis = 0) + np.std(rmse_vec_exp, axis = 0),
                     color='black', alpha=0.2)
    plt.xlabel('Latent space dimension')
    plt.ylabel('Covariance RMSE')
    tmp = "output\\rmse_vs_rank.pdf"
    filename = os.path.join(dirname, tmp)
    plt.tight_layout()
    plt.savefig(filename, quality = 95)
    plt.show()
    
    
    
if False: # Test likelihood comparison
    model = multimodal_gen(P = P, M = M, K = k, step_comp_elbo = 200, covariates = None, D = 0, full_dim =  True)
    model.fit([Xm], epoch = 2000)

    
    

# #############################################################################
# Estimate the covariance
# MultPCA
model = multimodal_gen(P = P, M = M, K = K, step_comp_elbo = 200, covariates = None, D = 0, full_dim =  True)
model.fit([Xm], epoch = 2000)
elbo = model.compute_test_elbo([Xm_te])

m_covo = model.S_ind
m_preco = linalg.inv(m_covo)
m_cov, m_prec = norm_cov(m_covo, m_preco)


# Emprical
emp_cov = np.dot(X.T, X) / N
model.comps[0].gauss_loglik(X_te, 0, np.linalg.inv(emp_cov))

# Glasso
modelg = GraphicalLassoCV()
#Xc = copula_transform((np.log(X*s + m + 1e-6) - np.log(X*s + m + 1e-6).mean(axis = 0)) / np.log(X*s + m + 1e-6).std(axis = 0))
Xc = copula_transform(X)
modelg.fit(Xc)
cov_ = modelg.covariance_
prec_ = modelg.precision_

# Ledoit
lw_cov_, _ = ledoit_wolf(X)
lw_prec_ = linalg.inv(lw_cov_)

# FA
model_fa = FactorAnalysis(n_components=3, random_state=0)
model_fa.fit(X)
fa_prec_ = model_fa.get_precision()
fa_cov_ = model_fa.get_covariance()



# #############################################################################
# Plot the results
# plot the covariances
covs = [
    ("Empirical", emp_cov),
    ("Ledoit-Wolf", lw_cov_),
    ("FA", fa_cov_),
    ("GraphicalLasso", cov_),
    ("MM", m_covo),
    ("True", cov),
]

vmax = cov.max()
for i, (name, this_cov) in enumerate(covs):
    plt.imshow(
        this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r
    )
    plt.xticks(())
    plt.yticks(())
    tmp = "output\\sim_single_com_cov_" + name + ".pdf"
    filename = os.path.join(dirname, tmp)
    plt.savefig(filename, quality = 95)
    

# plot the precisions
precs = [
    ("Empirical", linalg.inv(emp_cov)),
    ("Ledoit-Wolf", lw_prec_),
    ("FA", fa_prec_),
    ("GraphicalLasso", prec_),
    ("MM", m_preco),
    ("True", prec),
]
vmax = prec.max()
for i, (name, this_prec) in enumerate(precs):
    plt.imshow(
        np.ma.masked_equal(this_prec, 0),
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
        cmap=plt.cm.RdBu_r,
    )
    plt.xticks(())
    plt.yticks(())
    tmp = "output\\sim_single_com_prec_" + name + ".pdf"
    filename = os.path.join(dirname, tmp)
    plt.savefig(filename, quality = 95)
    




if True:
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)
    
    # plot the covariances
    covs = [
        ("Empirical", emp_cov),
        ("Ledoit-Wolf", lw_cov_),
        ("FA", fa_cov_),
        ("GraphicalLassoCV", cov_),
        ("MM", m_cov),
        ("True", cov),
    ]
    
    vmax = cov.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.subplot(2, 6, i + 1)
        plt.imshow(
            this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r
        )
        plt.xticks(())
        plt.yticks(())
        plt.title("%s cov" % name)
    
    
    # plot the precisions
    precs = [
        ("Empirical", linalg.inv(emp_cov)),
        ("Ledoit-Wolf", lw_prec_),
        ("FA", fa_prec_),
        ("GraphicalLasso", prec_),
        ("MM", m_prec),
        ("True", prec),
    ]
    vmax = prec.max()
    for i, (name, this_prec) in enumerate(precs):
        ax = plt.subplot(2, 6, i + 7)
        plt.imshow(
            np.ma.masked_equal(this_prec, 0),
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
            cmap=plt.cm.RdBu_r,
        )
        plt.xticks(())
        plt.yticks(())
        plt.title("%s prec" % name)
        if hasattr(ax, "set_facecolor"):
            ax.set_facecolor(".7")
        else:
            ax.set_axis_bgcolor(".7")
            
    dirname = os.getcwd()
    filename = os.path.join(dirname, "output\\sim_single_com.eps")
    plt.savefig(filename, quality = 95)
    


