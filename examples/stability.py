import argparse
import torch
import sys,os
import random
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.import_modules import import_attr
from models.vae_models import *
from utils.load_save import save_dictionary, load_dictionary
from examples.config_file import config
from utils.name_file import name_turbo

ig = config()
parser = argparse.ArgumentParser(description='Testing Weights')
parser.add_argument('--seed', type=int, default = 123, help='Seed number')
args = parser.parse_args()

threshold = 20.

dict_input = load_dictionary(ig.data_samples_dir+'stability.p')
YY_ = dict_input['Yfeature']
Y__ = YY_[YY_ < 100]
print(len(YY_), len(Y__))
input('enter')

arr = os.listdir(ig.data_samples_dir)
threshold = 100.
for i in arr:
    if 'turbo' in i and 'no' in i and str(args.seed) in i:
       dict_input = load_dictionary(ig.data_samples_dir+i)
       print(i)
       X_ = dict_input['Xepisodes']
       Y_ = dict_input['Yepisodes'].reshape(-1,)
       if Y_[Y_<threshold] != []:
          Xtrain = X_[(Y_<threshold),:]
          Ytrain = Y_[(Y_<threshold)]
       break

'''
filename = ig.data_samples_dir + name_turbo() + 'seed_' + str(args.seed) + '.p'
dict_input = load_dictionary(filename)
X_ = dict_input['Xepisodes']
Y_ = dict_input['Yepisodes'].reshape(-1,)
if Y_[Y_<threshold] != []:
   Xtrain = X_[(Y_<threshold),:]
   Ytrain = Y_[(Y_<threshold)].reshape(-1,1)
'''
print(Xtrain.shape,Y_.shape)
input('enter')

if ig.ea_type == 'vae':
   vae = VariationalAutoencoder(ig.high_dim, ig.low_dim).to(ig.device) # GPU
elif ig.ea_type == 'cvae':
   vae = CondVariationalAutoencoder(ig.high_dim, ig.low_dim).to(ig.device) # GPU
elif ig.ea_type == 'ae':
   vae = Autoencoder(ig.high_dim, ig.low_dim).to(ig.device) # GPU

path_n = ig.model_checkpoint+ig.ea_type+ '_encoder'+'_in'+str(ig.high_dim)+'_'+'proj'+str(ig.low_dim)+'.pth'
vae.encoder.load_state_dict(torch.load(path_n))
path_n = ig.model_checkpoint+ig.ea_type+ '_decoder'+'_in'+str(ig.high_dim)+'_'+'proj'+str(ig.low_dim)+'.pth'
vae.decoder.load_state_dict(torch.load(path_n))

inputs  = torch.Tensor(Xtrain).to(ig.device)
codes   = vae.encoder(inputs)
outputs = vae.decoder(codes).cpu().detach().numpy()
codes   = codes.cpu().detach().numpy()

obj_attr = import_attr(ig.path_to_tasks, attribute=ig.obj)
objective = obj_attr(seed=args.seed)

Yout = []
i  = 0
for x_copy in codes:
    i += 1
    o = objective.__call__(x_copy)
    Yout.append(o)

dict_out = OrderedDict([('Y', np.array(Ytrain)),('Yfeature', np.array(Yout))])
save_dictionary(ig.data_samples_dir+'stability.p', dict_out)

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['xtick.labelsize'] = 26
matplotlib.rcParams['ytick.labelsize'] = 26
matplotlib.rcParams['axes.labelsize']  = 29
matplotlib.rcParams['figure.facecolor']= 'white'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 26}

matplotlib.rc('font', **font)
plt.rc('axes', labelsize=26)

markersize = 16.0
fontsize   = 36.0
linewidth  = 6.0
fontweight = 'bold'


plt.figure(10, figsize=(15,10))
#plt.ylim([98,102])
plt.plot(Ytrain, '+', markersize=markersize, linewidth=linewidth, label='Orig')
plt.plot(Yout,   'o', markersize=markersize, linewidth=linewidth, label='Trans')
plt.ylabel('Cost', fontsize=fontsize, fontweight=fontweight)
plt.xlabel('Iteration', fontsize=fontsize, fontweight=fontweight)
plt.title(ig.motion.capitalize(), fontweight=fontweight)
plt.tight_layout()
plt.legend()
plt.savefig(ig.figs_dir + 'cost_compare.png')

plt.figure(11, figsize=(15,10))
plt.plot(Ytrain, Yout, '+', markersize=markersize, linewidth=linewidth)
plt.ylabel('Transformed Cost', fontsize=fontsize, fontweight=fontweight)
plt.xlabel('Original Cost', fontsize=fontsize, fontweight=fontweight)
plt.title(ig.motion.capitalize(), fontweight=fontweight)
plt.tight_layout()
plt.savefig(ig.figs_dir + 'cost_point_compare.png')


