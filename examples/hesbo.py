import GPy
import pyro
import pyro.contrib.gp as gp
import math
import torch
from pyDOE import lhs
from tqdm import tqdm
from scipy.stats import norm
from torch.distributions.normal import Normal
import argparse
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))

import numpy as np
from tfbo.utils.load_save import save_dictionary
from tfbo.utils.import_modules import import_attr
from tfbo.utils.name_file import name_turbo
from examples.config_file import config
from collections import OrderedDict
import random

parser = argparse.ArgumentParser(description='Testing Weights')
parser.add_argument('--seed', type=int, default = 123, help='Seed number')
args = parser.parse_args()
ig = config()

random.seed(a=args.seed)
obj_attr = import_attr(ig.path_to_tasks, attribute=ig.obj)
f = obj_attr(seed=args.seed)

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


def back_projection(low_obs, high_to_low, sign, device='cpu'):
    if len(low_obs.shape)==1:
        low_obs=low_obs.reshape((1, low_obs.shape[0]))
    n=low_obs.shape[0]
    high_dim=high_to_low.shape[0]
    low_dim=low_obs.shape[1]
    high_obs=torch.zeros((n,high_dim)).to(device)
    scale=1
    for i in range(high_dim):
        high_obs[:,i]=sign[i]*low_obs[:,high_to_low[i]]*scale

#    high_obs = torch.where(high_obs >  bx_size,  torch.Tensor([ bx_size]).to(device), high_obs)
#    high_obs = torch.where(high_obs < -bx_size,  torch.Tensor([-bx_size]).to(device), high_obs)
    return (high_obs/2+0.5)

def RunMain(f, seed):
    ARD=False
    variance=1.
    length_scale=None
    f_s = None
    max_V = 1
    min_V = 0
    j = 0
    if ig.method == 'hesbo':
       active_var= np.arange(ig.high_dim)
       high_to_low=torch.from_numpy(np.random.choice(range(ig.low_dim), ig.high_dim)).to(ig.device)
       sign = torch.from_numpy(np.random.choice([-1, 1], ig.high_dim)).to(ig.device)
       np.save(ig.model_checkpoint+'/hesbo/high_to_low_'+str(ig.low_dim)+'_'+str(ig.high_dim)+'_'+str(seed)+'.npy', high_to_low)
       np.save(ig.model_checkpoint+'/hesbo/sign_'+str(ig.low_dim)+'_'+str(ig.high_dim)+'_'+str(seed)+'.npy', sign)
    else:
       ig.low_dim = ig.high_dim
    while j < ig.n_init:
          x=lhs(ig.low_dim, 1) * (max_V-min_V) + min_V
          if ig.method == 'hesbo':
             xb = back_projection(x,high_to_low,sign,device=ig.device)
             fs = f.__call__(xb)
          else:
             fs = f.__call__(x)
          if not np.isnan(fs):
             if j==0:
                f_s = fs
                s = x
             else:
                s = np.vstack((s, x))
                f_s = np.vstack((f_s, fs))
             j += 1
    # Building and fitting a new GP model
    kern = GPy.kern.Matern52(input_dim=ig.low_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(s, f_s, kernel=kern)
    m.likelihood.variance = 1e-3
    update=True
    for i in tqdm(range(ig.max_evals)):
        xl = next_candidate(m, s, f_s, ig.low_dim, min_V, max_V, ig.method, update)
        if ig.method == 'hesbo':
           xh=back_projection(xl, high_to_low, sign, device=ig.device)
           fs = f.__call__(xh)
        else:
           fs = f.__call__(xl)
        if not np.isnan(fs):
           s   = np.vstack((s, xl))
           f_s = np.vstack((f_s, fs))
        if (i % 10) == 0:
           np.save(ig.model_checkpoint+'/hesbo/'+ig.motion+'_'+ig.method+'_costWeights_'+str(ig.low_dim)+'_'+str(ig.high_dim)+'_'+str(seed)+'.npy', s)
           np.save(ig.model_checkpoint+'/hesbo/'+ig.motion+'_'+ig.method+'_f_s_'+str(ig.low_dim)+'_'+str(ig.high_dim)+'_'+str(seed)+'.npy', f_s)


RunMain(f, seed=args.seed)
