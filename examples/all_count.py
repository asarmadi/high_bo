import GPy
import pyro
import pyro.contrib.gp as gp
import math
import numpy as np
import torch
from pyDOE import lhs
from tqdm import tqdm
from bo_cost import dist_func
from scipy.stats import norm
from torch.distributions.normal import Normal

def EI(D_size,f_max,mu,var):
    ei=np.zeros((D_size,1))
    std_dev=np.sqrt(var)
    kappa=0.0
    for i in range(D_size):
        if var[i]!=0:
            improve = mu[i] - f_max + kappa
            z= improve / std_dev[i]
            ei[i]= improve * norm.cdf(z) + std_dev[i] * norm.pdf(z)
    return ei

def next_candidate(m, X, Y, n_dim, min_val, max_val, method, update=True):
    if method == 'pure' or method == 'hesbo':
       m.set_XY(X, Y)
       if update:
          m.optimize()
       D = lhs(n_dim, 2000) * (max_val-min_val) + min_val
       mu, var = m.predict(D)
       ei_d = EI(len(D), max(Y), mu, var)
       index = np.argmax(ei_d)
       return D[index]
    elif method == 'random':
       return lhs(n_dim, 1) * (max_val-min_val) + min_val


def back_projection(low_obs, high_to_low, sign, bx_size, device):
    if len(low_obs.shape)==1:
        low_obs=low_obs.reshape((1, low_obs.shape[0]))
    n=low_obs.shape[0]
    high_dim=high_to_low.shape[0]
    low_dim=low_obs.shape[1]
    high_obs=torch.zeros((n,high_dim)).to(device)
    scale=1
    for i in range(high_dim):
        high_obs[:,i]=sign[i]*low_obs[:,high_to_low[i]]*scale

    high_obs = torch.where(high_obs >  bx_size,  torch.Tensor([ bx_size]).to(device), high_obs)
    high_obs = torch.where(high_obs < -bx_size,  torch.Tensor([-bx_size]).to(device), high_obs)

    return high_obs

def RunMain(low_dim=2, high_dim=25, n_initial=10, total_itr=100, active_var=None, ARD=False, variance=1., length_scale=None,
            box_size=None, hyper_opt_interval=20, noise_var=0, device='cpu', file_name='./output_weights', method='pure', motion='trot', seed_num=56):

    f_s = None
    max_V = 1e+4
    min_V = 0
    j = 0
    if method == 'hesbo':
       active_var= np.arange(high_dim)
       if box_size is None:
          box_size=1
       max_V = box_size
       min_V = -box_size
       high_to_low=torch.from_numpy(np.random.choice(range(low_dim), high_dim)).to(device)
       sign = torch.from_numpy(np.random.choice([-1, 1], high_dim)).to(device)
       np.save(file_name+motion+'/hesbo/high_to_low_'+str(low_dim)+'_'+str(high_dim)+'_'+str(seed_num)+'.npy', high_to_low)
       np.save(file_name+motion+'/hesbo/sign_'+str(low_dim)+'_'+str(high_dim)+'_'+str(seed_num)+'.npy', sign)
    else:
       low_dim = high_dim
    while j < n_initial:
          test_func = dist_func(bx_size=box_size,motion=motion, method=method, n_high=high_dim)
          x=lhs(low_dim, 1) * (max_V-min_V) + min_V
          if method == 'hesbo':
             xb = back_projection(x,high_to_low,sign,box_size, device)
             fs = test_func.evaluate(xb)
          else:
             fs = test_func.evaluate(x)
          if not np.isnan(fs):
             if j==0:
                f_s = fs
                s = x
             else:
                s = np.vstack((s, x))
                f_s = np.vstack((f_s, fs))
             j += 1
          del test_func
    # Building and fitting a new GP model
    kern = GPy.kern.Matern52(input_dim=low_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(s, f_s, kernel=kern)
    m.likelihood.variance = 1e-3
    
    for i in tqdm(range(total_itr)):
        test_func = dist_func(bx_size=box_size,motion=motion, method=method, n_high=high_dim)
        if method == 'hesbo':
           update=True
#           if (i+n_initial<=25 and i % 5 == 0) or (i+n_initial>25 and i % hyper_opt_interval == 0):
#              update=True
#           else:
#              update=False
        else:
           update=True
        xl = next_candidate(m, s, f_s, low_dim, min_V, max_V, method, update)
        if method == 'hesbo':
           xh=back_projection(xl, high_to_low, sign, box_size, device)
           fs = test_func.evaluate(xh)
        else:
           fs = test_func.evaluate(xl)
        if not np.isnan(fs):
           s   = np.vstack((s, xl))
           f_s = np.vstack((f_s, fs))
        if (i % 10) == 0:
           np.save(file_name+motion+'/'+method+'/costWeights_'+str(low_dim)+'_'+str(high_dim)+'_'+str(seed_num)+'.npy', s)
           np.save(file_name+motion+'/'+method+'/f_s_'+str(low_dim)+'_'+str(high_dim)+'_'+str(seed_num)+'.npy', f_s)
        del test_func
