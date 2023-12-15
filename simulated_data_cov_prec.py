import numpy as np
from scipy import linalg
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import make_sparse_spd_matrix
import matplotlib.pyplot as plt
from simulated_data_exps import data_gen
import os
import sys
import copy
from os.path import dirname
from multimodal_gen import multimodal_gen
from GemBag import GemBag_py
from jSDM import jSDM_py
from copula import copula_transform
from scipy.special import softmax
sys.path.insert(1, os.path.join(dirname(os.getcwd()), 'Library'))
from rpy2.robjects import numpy2ri,IntVector,Formula


dirname = os.getcwd()

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



# Simulation variables
covariates = False
full_dim = True
sparse = False
data_gen_mm = True

P = [2]  # of modalities
P += [0]
K = 5  # Latent space dimension
L = 2 # Environment
N = 200
D = 0
M = [20, 10] # Observation dimension

res_vec_cov = []
res_vec_prec = []
for e in range(1):

    if data_gen_mm:
        # Generate multpca data
        C_max = np.random.randint(1000, 2000, P[0]) # Total number of counts - 1 (if 0 -> Categorical)
        Xlm = []
        W_lm = []
        for l in range(L):
            Xm, _, _, _, W, _, _ = data_gen(K, P[0], N, M, C_max, full_dim, covariates, D)
            Xlm.append(Xm)
            W_lm.append(W)
    else:
        # Generate jsdm data
        x1 = np.repeat(np.arange(L), int(N*0.8))
        x = np.vstack((np.ones(int(N*0.8*L)),x1)).T
        z = np.array([np.random.multivariate_normal(np.zeros(K), np.eye(K)) for i in range(int(N*0.8*L))])
        V = np.random.multivariate_normal(np.zeros(K), np.eye(K), size = sum(np.array(M)+1))
        B = np.random.multivariate_normal(np.zeros(L), np.eye(L), size = sum(np.array(M)+1))
        mu = np.exp(z @ V.T + x @ B.T)
        X = np.array([np.random.poisson(mu[i]) for i in range(len(mu))])
        X = np.split(X, L)
        Xlm = [[x[:,:M[0]+1], x[:, M[0]+1:]] for x in X]
        for i in range(len(Xlm)):
            for j in range(len(Xlm[i])):
                Xlm[i][j] = np.concatenate([Xlm[i][j][:, :-1], Xlm[i][j].sum(axis = 1)[None].T], axis = 1)


    # Data for other algorithms
    Xlo = []
    for l in range(L):
        X = []
        for i in range(P[0]):
            Ni = Xlm[l][i][:, -1]
            X.append(np.append(Xlm[l][i][:, :-1], (Ni - np.sum(Xlm[l][i][:, :-1], axis = 1))[None].T, axis = 1))
        Xlo.append(X)
    
    Xlo = [np.hstack(X) for X in Xlo]
    Xl = copy.deepcopy(Xlo)
    for X in Xl:
        X = X.astype(float)
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
    
    # True covariance
    if data_gen_mm:
        Wind = np.vstack(W_lm[0])
        Ai = np.zeros((Wind.shape[0], Wind.shape[0]))
        cnt = 0
        for p in range(P[0]):
            A = 0.5*(np.eye(M[p]+1) - (1/(M[p]+2))*np.ones((M[p]+1, M[p]+1)))
            Ai[cnt:cnt + A.shape[0], cnt:cnt + A.shape[0]] = np.linalg.inv(A)
            cnt += A.shape[0]
        covo = Wind @ Wind.T + Ai
    else:
        covo = V @ V.T + np.eye(V.shape[0])

    preco = linalg.inv(covo)
    cov, prec = norm_cov(covo, preco)
    
    
    # Estimate the covariance
    # JSDM
    u1 = np.repeat(np.arange(L), int(N*0.8))
    U = np.vstack((np.ones(int(N*0.8*L)),u1)).T
    fmla = Formula("~u1") # R formula, include independent variables
    env = fmla.environment
    env['u1'] = u1
    jsdm_covo = jSDM_py(np.vstack(Xlo),U,fmla, n_latent=K) # 2 covariane of size d by d returned
    jsdm_preco = linalg.inv(jsdm_covo)
    jsdm_cov, jsdm_prec = norm_cov(jsdm_covo, jsdm_preco)


    # Gembag
    v0_l = np.array([0.25, 0.5, 0.75, 1]) * np.sqrt(1/N/np.log(sum(np.array(M)+1)))
    v1_l = np.array([2.5, 5, 7.5, 10]) * np.sqrt(1/N/np.log(sum(np.array(M)+1)))
    gem_covs = GemBag_py([X.T for X in Xl], L, np.array([N]*L), v0_l, v1_l)
    gem_covo = gem_covs[0]
    gem_preco = linalg.inv(gem_covo)
    gem_cov, gem_prec = norm_cov(gem_covo, gem_preco)


    # MultPCA
    Xm = [Xlm[0][i].astype(np.float64) for i in range(P[0])]
    model = multimodal_gen(P = P, M = M, K = K, D = 0, full_dim =  True)
    model.fit(Xm, epoch = 100000, step_comp_elbo = 10000)
    m_covo = model.S_ind
    m_preco = linalg.inv(m_covo)
    m_cov, m_prec = norm_cov(m_covo, m_preco)
    
    # Emprical
    emp_covo = np.dot(X.T, X) / N
    emp_preco = np.linalg.inv(emp_covo)
    emp_cov, emp_prec = norm_cov(emp_covo, emp_preco)
    
    # Glasso
    modelg = GraphicalLassoCV()
    Xc = copula_transform(X)
    modelg.fit(Xc)
    g_covo = modelg.covariance_
    g_preco = modelg.precision_
    g_cov, g_prec = norm_cov(g_covo, g_preco)
    
    # Ledoit
    lw_covo, _ = ledoit_wolf(X)
    lw_preco = linalg.inv(lw_covo)
    lw_cov, lw_prec = norm_cov(lw_covo, lw_preco)
    
    # FA
    model_fa = FactorAnalysis(n_components=K, random_state=0)
    model_fa.fit(X)
    fa_preco = model_fa.get_precision()
    fa_covo = model_fa.get_covariance()
    fa_cov, fa_prec = norm_cov(fa_covo, fa_preco)
    
    res_vec_cov.append([
        np.sqrt(np.mean((emp_cov - cov)**2)),
        np.sqrt(np.mean((lw_cov - cov)**2)),
        np.sqrt(np.mean((fa_cov - cov)**2)),
        np.sqrt(np.mean((g_cov - cov)**2)),
        np.sqrt(np.mean((gem_cov - cov)**2)),
        np.sqrt(np.mean((jsdm_cov - cov)**2)),
        np.sqrt(np.mean((m_cov - cov)**2))]
        )
    res_vec_prec.append([
        
        np.sqrt(np.mean((emp_prec - prec)**2)),
        np.sqrt(np.mean((lw_prec - prec)**2)),
        np.sqrt(np.mean((fa_prec - prec)**2)),
        np.sqrt(np.mean((g_prec - prec)**2)),
        np.sqrt(np.mean((gem_prec - prec)**2)),
        np.sqrt(np.mean((jsdm_prec - prec)**2)),
        np.sqrt(np.mean((m_prec - prec)**2))]
        )

res_vec_cov = np.array(res_vec_cov)
res_vec_prec = np.array(res_vec_prec)

# MSE results
print('Mean Cov')
[print(np.mean(res_vec_cov[:, i])) for i in range(res_vec_cov.shape[-1])]
print('Std Cov')
[print(np.std(res_vec_cov[:, i])) for i in range(res_vec_cov.shape[-1])]

print('Mean Prec')
[print(np.mean(res_vec_prec[:, i])) for i in range(res_vec_prec.shape[-1])]
print('Std Prec')
[print(np.std(res_vec_prec[:, i])) for i in range(res_vec_prec.shape[-1])]



if True:
    # Plot the results for saving
    # plot the covariances
    covs = [
        ("Empirical", emp_cov),
        ("Ledoit-Wolf", lw_cov),
        ("GraphicalLasso", g_cov),
        ("Gembag", gem_cov),
        ("FA", fa_cov),
        ("jsdm", jsdm_cov),
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
        plt.colorbar()
        tmp = "output/sim_cov_" + str(covariates) + "_" + str(P[0]) + "_" + name + ".pdf"
        filename = os.path.join(dirname, tmp)
        plt.savefig(filename)
        plt.show()
        
    
    # plot the precisions
    precs = [
        ("Empirical", linalg.inv(emp_cov)),
        ("Ledoit-Wolf", lw_prec),
        ("GraphicalLasso", jsdm_prec),
        ("Gembag", gem_prec),
        ("FA", fa_prec),
        ("jsdm", jsdm_prec),
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
        plt.colorbar()
        plt.xticks(())
        plt.yticks(())
        tmp = "output/sim_prec_"  + str(covariates) + "_" + str(P[0]) + "_" + name + ".pdf"
        filename = os.path.join(dirname, tmp)
        plt.savefig(filename)
        plt.show()
    
    
    
    # Plot the results for visualization
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.02, right=0.98)
    
    # plot the covariances
    covs = [
        ("Empirical", emp_cov),
        ("Ledoit-Wolf", lw_cov),
        ("GraphicalLasso", g_cov),
        ("Gembag", gem_cov),
        ("FA", fa_cov),
        ("jsdm", jsdm_cov),
        ("MM", m_cov),
        ("True", cov),
    ]
    vmax = cov.max()
    for i, (name, this_cov) in enumerate(covs):
        plt.subplot(2, 8, i + 1)
        plt.imshow(
            this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r
        )
        plt.xticks(())
        plt.yticks(())
        plt.title("%s cov" % name)
    
    
    # plot the precisions
    precs = [
        ("Empirical", linalg.inv(emp_cov)),
        ("Ledoit-Wolf", lw_prec),
        ("GraphicalLasso", jsdm_prec),
        ("Gembag", gem_prec),
        ("FA", fa_prec),
        ("jsdm", jsdm_prec),
        ("MM", m_prec),
        ("True", prec),
    ]
    vmax = prec.max()
    for i, (name, this_prec) in enumerate(precs):
        ax = plt.subplot(2, 8, i + 9)
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
        #plt.savefig(filename)
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
        plt.savefig(filename)
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
        plt.savefig(filename)
        plt.show()
        