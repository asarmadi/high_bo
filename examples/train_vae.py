import torch
import sys,os
import random
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from collections import OrderedDict

import numpy as np
from utils.import_modules import import_attr
from models.vae_models import Autoencoder, VariationalAutoencoder, CondVariationalAutoencoder
from utils.load_save import save_dictionary, load_dictionary
from torch.optim.lr_scheduler import MultiStepLR
from examples.config_file import config

ig = config()

def train(autoencoder, data, labels=None, ig=None):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=ig.lr)
    scheduler = MultiStepLR(opt, milestones=[1700,1900], gamma=0.1)
    for epoch in range(ig.n_epochs):
        total_loss = 0
        for i in range(0,len(data),ig.bs):
            x = torch.Tensor(data[i:i+ig.bs]).to(ig.device) # GPU
            opt.zero_grad()
            if ig.ea_type == 'cvae':
               y = torch.Tensor(labels[i:i+ig.bs]).to(ig.device)
               x_hat = autoencoder(x, y/5100.)
            else:
               x_hat = autoencoder(x)
            if ig.ea_type == 'ae':
               loss = ((x - x_hat)**2).sum()
            else:
               loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print("Epoch:{}/{}, Loss:{}".format(epoch,ig.n_epochs,total_loss/len(data)))
        if epoch % 100 == 0:
           torch.save(autoencoder.state_dict(), ig.model_checkpoint + ig.ea_type+'_encoder_in'+str(ig.high_dim)+'_'+'proj'+str(ig.low_dim)+'.pth')
           torch.save(autoencoder.state_dict(), ig.model_checkpoint + ig.ea_type+'_decoder_in'+str(ig.high_dim)+'_'+'proj'+str(ig.low_dim)+'.pth')
#        scheduler.step(epoch)
    return autoencoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

Xtrain = np.zeros([0, ig.high_dim])
Ytrain = np.zeros([0, 1])
arr = os.listdir(ig.data_samples_dir)
threshold = 100.
k = 0
for i in arr:
    if 'turbo' in i and 'no' in i and k < 5:
       dict_input = load_dictionary(ig.data_samples_dir+i)
       print(i)
       X_ = dict_input['Xepisodes']
       Y_ = dict_input['Yepisodes'].reshape(-1,)
       if Y_[Y_<threshold] != []:
          Xtrain = np.append(Xtrain, X_[(Y_<threshold),:], axis=0)
          Ytrain = np.append(Ytrain, Y_[(Y_<threshold)].reshape(-1,1), axis=0)
       k += 2



if ig.ea_type == 'vae':
   vae = VariationalAutoencoder(ig.high_dim, ig.low_dim).to(ig.device) # GPU
elif ig.ea_type == 'cvae':
   vae = CondVariationalAutoencoder(ig.high_dim, ig.low_dim).to(ig.device) # GPU
elif ig.ea_type == 'ae':
   vae = Autoencoder(ig.high_dim, ig.low_dim).to(ig.device) # GPU

print(Ytrain.shape, count_parameters(vae))
input('enter')

vae = train(vae, Xtrain, Ytrain, ig)
torch.save(vae.encoder.state_dict(), ig.model_checkpoint + ig.ea_type+'_encoder_in'+str(ig.high_dim)+'_'+'proj'+str(ig.low_dim)+'.pth')
torch.save(vae.decoder.state_dict(), ig.model_checkpoint + ig.ea_type+'_decoder_in'+str(ig.high_dim)+'_'+'proj'+str(ig.low_dim)+'.pth')

sample = torch.Tensor(Xtrain[:1]).to(ig.device)
if ig.ea_type == 'vae':
   print(sample.data,vae(sample))
elif ig.ea_type == 'cvae':
   Y = torch.Tensor([[0]])
   print(sample.data,vae(sample, Y))

