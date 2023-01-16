import numpy as np
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from tfbo.utils.import_modules import import_attr
from tfbo.utils.name_file import name_synthetic, name_turbo
from tfbo.utils.load_save import load_dictionary
from examples.config_file import config
import matplotlib.pyplot as plt

ig = config()

if ig.method == 'turbo':
   filename = name_turbo()
obj_attr = import_attr(ig.path_to_tasks, attribute=ig.obj)
objective = obj_attr()
arr = os.listdir(ig.data_samples_dir)
fs_list = []
for i in arr:
    if 'turbo' in i:
       dict_input = load_dictionary(ig.data_samples_dir+i)
       X_ = dict_input['Xepisodes']
       Y_ = dict_input['Yepisodes'].reshape(-1,)
       if ig.method == 'turbo':
          x_copy = X_[np.argmin(Y_),:]
       else:
          x_copy = X_[0,0,np.argmin(Y_),:]
       x_h, fs, o = objective.f(x_copy, eval=True)
    fs_list.append(fs[0][0])

print(fs_list)
print(np.mean(fs_list), np.std(fs_list))

plt.figure(10)
plt.plot(fs_list, '+')
plt.ylabel("Cost Value")
plt.savefig(ig.figs_dir + 'all_files.png')

