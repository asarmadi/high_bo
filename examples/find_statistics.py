import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
import time
import torch
from utils.import_modules import import_attr
from utils.name_file import name_synthetic, name_turbo
from utils.load_save import load_dictionary
from examples.config_file import config

parser = argparse.ArgumentParser(description='Testing Weights')
parser.add_argument('--animation', action='store_true', default=False, help='Generate animation')
parser.add_argument('--seed', type=int, default = 0, help='Seed number')
args = parser.parse_args()

ig = config()

if ig.method == 'turbo':
   filename = name_turbo()
#filename='cma_'
obj_attr  = import_attr(ig.path_to_tasks, attribute=ig.obj)
objective = obj_attr(seed=args.seed)
#path_dir = ig.data_samples_dir + filename + 'seed_'+str(args.seed) + 'eaType_' + ig.ea_type + '.p'
#path_dir = ig.data_samples_dir + 'cma_seed_'+str(args.seed) + '.p'
if ig.method != 'pre':
   path_dir = ig.data_samples_dir + filename + 'seed_'+str(args.seed) + '.p'
   print(path_dir)
   input('enter')
   dict_input = load_dictionary(path_dir)
   X_ = dict_input['Xepisodes']
   Y_ = dict_input['Yepisodes']
#   idx = (~np.isnan(Y_)).reshape(-1,)
#   print(idx.shape)
#   X_ = X_[idx,:]
#   Y_ = Y_[idx,:]
   if ig.method == 'turbo':
      x_copy = X_[np.argmin(Y_),:]
      print(x_copy, np.min(Y_))
      input('enter')
   else:
      x_copy = X_[0,0,np.argmin(Y_),:]
   if args.animation:
      ig.show_gui = True
      x_h, fs, o = objective.f(x_copy, record_file_name=ig.figs_dir+ig.motion+'_'+ig.method+'_'+str(args.seed)+'.mp4', eval=True)
   else:
      x_h, fs, o = objective.f(x_copy, eval=True)
else:
   ig.show_gui = False
   x_h, fs, o = objective.f([], record_file_name=ig.figs_dir+ig.motion+'_predefined.mp4', eval=True)

print(fs, o)

