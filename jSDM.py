from numpy.random import default_rng
import numpy as np

import rpy2

from rpy2.robjects import numpy2ri,IntVector,Formula
from rpy2.robjects.vectors import ListVector
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

numpy2ri.activate()

utils = importr('utils')
r = robjects.r
jSDM = importr('jSDM')

def jSDM_py(Y,X,fmla, n_latent=2,mcmc=10000, burnin=5000, thin=10):
  '''
  Inputs
    Y: count data (number of samples by number of features)
    X: enviromental variables (number of samples by number of enviromental varibles)
      e.g binary varibles indicate presence of wild type
      note: this should include a constant 1 vector to represent intercept at it should be the first feature
    fmla: R-formula format specified the model, e.g ~x1+x2 means use x1 and x2 to regress on the target variable
    n_latent: number of latent variables
    mcmc: number of mcmc samples
    burn-in: number of burn-in samples, note total samples= burn-in + mcmc
    thin:  how much the mcmc samples should be thined, e.g thin=10 means only one sample out of 10 is kept
           i.e effective number of posterior samples = mcmc/thin

  Immediate
    beta_coef: posterior samples of regression coefficients (correspondes to envirioments)

  Outputs
    covs: covairance matrix
  '''
  _,d = Y.shape
  K = X.shape[1]
  model = jSDM.jSDM_poisson_log(count_data=Y,site_data=X,site_formula=fmla,n_latent=n_latent,mcmc=mcmc,burnin=burnin,thin=thin)
  print(model) #make sure the index below correspond to mcmc.sp (posterior samples of regression coefficients)
  cov = jSDM.get_residual_cor(model)[5]

  ### Legacy code using regression coefficients
  # beta_coef = [np.empty((mcmc//thin,d)) for k in range(K-1)]
  # for j in range(d):
  #   for k in range(1,K):
  #     beta_coef[k-1][:,j] = model[0][j][:,k]
  # covs = [np.cov(beta_coef[k].T) for k in range(K-1)]
  return cov

if __name__ == "__main__":
  n = 60 # number of samples
  d = 20 # number of species (features)
  rng = default_rng(seed=1234)
  x1 = rng.normal(size=n)

  x2 = rng.normal(size=n)
  X = np.vstack((np.ones(n),x1,x2)).T
  beta = rng.multivariate_normal(mean=np.zeros(d),cov=np.eye(d),size=3)
  W = np.vstack((rng.normal(size=n),rng.normal(size=n))).T
  n_latent = W.shape[1]
  lam2 = 2*(rng.random((2,d))-0.5)
  Y = rng.poisson(lam=X@beta+W@lam2+6)
  # Y = rng.poisson(lam=100,size=(n,d))

  fmla = Formula("~x1+x2") # R formula, include independent variables
  env = fmla.environment
  env['x1'] = x1
  env['x2'] = x2
  cov = jSDM_py(Y,X,fmla,n_latent=2) # covariance of size d by d returned

