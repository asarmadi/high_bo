import torch
import sys,os
import random
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from collections import OrderedDict

import numpy as np
from tfbo.utils.import_modules import import_attr
from tfbo.models.vae_models import VariationalEncoder
from tfbo.utils.load_save import save_dictionary, load_dictionary
from examples.config_file import config
from tfbo.utils.name_file import name_turbo

ig = config()

Xtrain = np.zeros([0, ig.high_dim])
Ytrain = np.zeros([0, 1])
arr = os.listdir(ig.data_samples_dir)
threshold = 7.
for i in arr:
    if 'turbo' in i and 'no' in i:
       dict_input = load_dictionary(ig.data_samples_dir+i)
       X_ = dict_input['Xepisodes']
       Y_ = dict_input['Yepisodes'].reshape(-1,)
       if Y_[Y_<threshold] != []:
          Xtrain = np.append(Xtrain, X_[(Y_<threshold),:], axis=0)
          Ytrain = np.append(Ytrain, Y_[(Y_<threshold)].reshape(-1,1), axis=0)

if ig.ea_type != 'no':
   encoder = VariationalEncoder(ig.high_dim, ig.low_dim).to(ig.device) # GPU
   path_n = ig.model_checkpoint+ig.ea_type+ '_encoder'+'_in'+str(ig.high_dim)+'_'+'proj'+str(ig.low_dim)+'.pth'
   encoder.load_state_dict(torch.load(path_n))

print(Ytrain.shape)
input('enter')

inputs  = torch.Tensor(Xtrain).to(ig.device)
outputs = encoder(inputs)
print(outputs.shape)
input('enter')

filename = ig.data_samples_dir + name_turbo() + 'eaType_' + ig.ea_type  +'_init.p'
dict_out = OrderedDict([('Xepisodes', outputs),('Yepisodes', Ytrain)])
save_dictionary(filename, dict_out)


