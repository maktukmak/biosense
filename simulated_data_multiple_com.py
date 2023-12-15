import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt


from simulated_data_exps import data_gen
import os
import sys
from os.path import dirname
sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))
dirname = os.getcwd()

from multimodal_gen import multimodal_gen
from multPCA import multPCA

import pickle
import seaborn as sns

# Simulation variables
covariates = False
full_dim = True
P = [2]  # of modalities
P += [0]
K = 3  # Latent space dimension
N = 200
D = 2
M = [20, 10] # Observation dimension

res_vec_cov = []
res_vec_prec = []

for e in range(1):

    # #############################################################################
    # Generate the data
    C_max = np.random.randint(1000, 2000, P[0]) # Total number of counts - 1 (if 0 -> Categorical)
    
    Xm, X_te, prob, prob_te, W, U, U_te = data_gen(K, P[0], N, M, C_max, full_dim, covariates, D)
    Xm = [Xm[i].astype(np.float64) for i in range(P[0])]
    
    # True covariance
    Wind = np.vstack(W)
    Ai = np.zeros((Wind.shape[0], Wind.shape[0]))
    cnt = 0
    for p in range(P[0]):
        A = 0.5*(np.eye(M[p]+1) - (1/(M[p]+2))*np.ones((M[p]+1, M[p]+1)))
        Ai[cnt:cnt + A.shape[0], cnt:cnt + A.shape[0]] = np.linalg.inv(A)
        cnt += A.shape[0]

    covo = Wind @ Wind.T + Ai
    preco = linalg.inv(covo)
    d = np.sqrt(np.diag(covo))
    cov = covo / d
    cov /= d[:, np.newaxis]
    prec = preco * d
    prec *= d[:, np.newaxis]
    

    # Data for other algorithms
    X = []
    for i in range(P[0]):
        Ni = Xm[i][:, -1]
        X.append(np.append(Xm[i][:, :-1], (Ni - np.sum(Xm[i][:, :-1], axis = 1))[None].T, axis = 1))
    
    X = np.hstack(X)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    
    
    # #############################################################################
    # Estimate the covariance
    # MultPCA
    model = multimodal_gen(P = P, M = M, K = K, D = 0, full_dim =  True)
    model.fit(Xm, epoch = 100000, step_comp_elbo = 1000)
    
    m_covo = model.S_ind
    m_preco = linalg.inv(m_covo)
    
    if False:
        # Pooled MultPCA
        model = multimodal_gen(P = [1, 0], M = [sum(M)+1], K = K, step_comp_elbo = 10, covariates = None, D = 0, full_dim =  True)
        Xmp = []
        for i in range(len(P)):
            Ni = Xm[i][:, -1]
            Xmp.append(np.append(Xm[i][:, :-1], (Ni - np.sum(Xm[i][:, :-1], axis = 1))[None].T, axis = 1))
        Xmp = np.hstack(Xmp)
        Xmp[:, -1] = np.sum(Xmp, axis = 1)
        model.fit([Xmp], epoch = 500)
        m_covp = model.S_ind
        m_precp = linalg.inv(m_covp)
    
    # Conditional dependency
    Chh = m_covo[0:M[0], 0:M[0]]
    Cvv = m_covo[M[0]:, M[0]:]
    Cvh = m_covo[M[0]:, 0:M[0]]
    Chv = m_covo[0:M[0], M[0]:]
    m_cov_cond = Chh - Chv @ np.linalg.inv(Cvv) @ Cvh
    
    d = np.sqrt(np.diag(m_covo))
    m_cov = m_covo / d
    m_cov /= d[:, np.newaxis]
    m_prec = m_preco * d
    m_prec *= d[:, np.newaxis]
    
    # Emprical
    emp_cov = np.dot(X.T, X) / N
    emp_prec= np.linalg.inv(emp_cov)
    
    # Glasso
    modelg = GraphicalLassoCV()
    modelg.fit(X)
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
    
    res_vec_cov.append([
        np.sqrt(np.mean((m_cov - cov)**2)),
        np.sqrt(np.mean((emp_cov - cov)**2)),
        np.sqrt(np.mean((cov_ - cov)**2)),
        np.sqrt(np.mean((fa_cov_ - cov)**2)),
        np.sqrt(np.mean((lw_cov_ - cov)**2))]
        )
    res_vec_prec.append([
        np.sqrt(np.mean((m_prec - prec)**2)),
        np.sqrt(np.mean((emp_prec - prec)**2)),
        np.sqrt(np.mean((prec_ - prec)**2)),
        np.sqrt(np.mean((fa_prec_ - prec)**2)),
        np.sqrt(np.mean((lw_prec_ - prec)**2))]
        )

res_vec_cov = np.array(res_vec_cov)
res_vec_prec = np.array(res_vec_prec)

# MSE results
print('Mean Cov')
[print(np.mean(res_vec_cov[:, i])) for i in range(5)]
print('Std Cov')
[print(np.std(res_vec_cov[:, i])) for i in range(5)]

print('Mean Prec')
[print(np.mean(res_vec_prec[:, i])) for i in range(5)]
print('Std Prec')
[print(np.std(res_vec_prec[:, i])) for i in range(5)]



if True:
    # Plot the results
    # plot the covariances
    covs = [
        ("Empirical", emp_cov),
        ("Ledoit-Wolf", lw_cov_),
        ("GraphicalLasso", cov_),
        ("FA", fa_cov_),
        ("MM", m_cov),
        ("True", cov),
    ]
    
    vmax = cov.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.imshow(
            this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r
        )
        plt.xticks(())
        plt.yticks(())
        tmp = "output\\sim_cov_" + str(covariates) + "_" + str(P[0]) + "_" + name + ".pdf"
        filename = os.path.join(dirname, tmp)
        plt.savefig(filename)
        
    
    # plot the precisions
    precs = [
        ("Empirical", linalg.inv(emp_cov)),
        ("Ledoit-Wolf", lw_prec_),
        ("GraphicalLasso", prec_),
        ("FA", fa_prec_),
        ("MM", m_prec),
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
        tmp = "output\\sim_prec_"  + str(covariates) + "_" + str(P[0]) + "_" + name + ".pdf"
        filename = os.path.join(dirname, tmp)
        plt.savefig(filename)
    
    
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.02, right=0.98)
    
    # plot the covariances
    covs = [
        ("Empirical", emp_cov),
        ("Ledoit-Wolf", lw_cov_),
        ("GraphicalLassoCV", cov_),
        ("FA", fa_cov_),
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
        ("GraphicalLasso", prec_),
        ("FA", fa_prec_),
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
    filename = os.path.join(dirname, "output\\sim_multi_com.eps")
    plt.savefig(filename)
    

if False:
    # Plot the conditional vs marginal
    covs = [
        ("Marginal", Chh),
        ("Conditional", m_cov_cond),
    ]
    vmax = cov.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.imshow(
            this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r
        )
        plt.xticks(())
        plt.yticks(())
        tmp = "output\\sim_multi_com_cov_" + name + ".pdf"
        filename = os.path.join(dirname, tmp)
        #plt.savefig(filename, quality = 95)
        plt.show()

if False:
    # Plot the seperate ve pooled
    covs = [
        ("Seperate", m_covo),
        ("Pooled", m_covp),
        ("True", covo),
        
    ]
    vmax = covo.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.imshow(
            this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r
        )
        plt.xticks(())
        plt.yticks(())
        tmp = "output\\sim_multi_com_pooling_cov_" + name + ".pdf"
        filename = os.path.join(dirname, tmp)
        plt.savefig(filename, quality = 95)
        plt.show()
        
    # Plot the seperate ve pooled
    precs = [
        ("Seperate", m_preco),
        ("Pooled", m_precp),
        ("True", preco),
    ]
    vmax = preco.max()
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
        tmp = "output\\sim_multi_com_pooling_prec_" + name + ".pdf"
        filename = os.path.join(dirname, tmp)
        plt.savefig(filename, quality = 95)
        plt.show()
        