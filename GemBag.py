from numpy.random import default_rng
import numpy as np

import rpy2

from rpy2.robjects import numpy2ri, IntVector, Formula
from rpy2.robjects.vectors import ListVector
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

numpy2ri.activate()

utils = importr('utils')
rcpp = importr('Rcpp')
r = robjects.r

# Loading the packges
r['source']('functions/BIC_GemBag.R')
rcpp.sourceCpp('functions/GemBag-algo.cpp')

Tune_GemBag = robjects.globalenv["Tune_GemBag"]
GemBag = robjects.globalenv["GemBag"]

def GemBag_py(Y_l,k,n_l, v0_l, v1_l,max_iters=20):
  '''
  Inputs
    Y_l: list of abundance data of different groups
    k:   number of groups
    n_l: number of samples for each group
         WARNING-need to be numpy array not list
    v0_l,v1_l: grid for hyperparameters
    max_iers:  maximum number of iterations

  Immediate
    S_l: sample covariance matrix for each group

  Outputs
    covs: covairance matrix
  '''

  S_l=[]
  for i in range(k):
    S_l.append(np.cov(Y_l[i]))
  S_lr = ListVector([(i+1, s) for i, s in enumerate(S_l)])

  p1 = 0.5

  hyper = Tune_GemBag(v0_l, v1_l, S_lr, n_l, max_iters, p1, 1)
  v0=hyper.rx2("v0")
  v1=hyper.rx2("v1")
  res = GemBag(S_l=S_lr, n=n_l, v_0=v0, v_1=v1, tau=v0, p_1=p1, p_2=1, maxiter=max_iters)
  res.names=r.c('Theta', 'P', 'W')

  return list(res.rx2('W'))

if __name__ == "__main__":

  # simulating example dataset
  n=20
  p=10
  k=3
  n_l=np.array([n]*k)

  Y_l=[]
  S_l=[]
  rng = default_rng(1234)
  for i in range(k):
    Y_l.append(rng.multivariate_normal(np.zeros(p),np.eye(p),size=n))
  v0_l = np.array([0.25, 0.5, 0.75, 1]) * np.sqrt(1/n/np.log(p))
  v1_l = np.array([2.5, 5, 7.5, 10]) * np.sqrt(1/n/np.log(p))

  covs = GemBag_py(Y_l,k,n_l, v0_l, v1_l)
  for cov in covs:
    print(cov.shape)

