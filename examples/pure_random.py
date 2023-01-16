import argparse
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))

import numpy as np
from tfbo.utils.load_save import save_dictionary
from tfbo.utils.import_modules import import_attr
from examples.config_file import config
from collections import OrderedDict
import random

parser = argparse.ArgumentParser(description='Testing Weights')
parser.add_argument('--seed', type=int, default = 123, help='Seed number')
args = parser.parse_args()
ig = config()

random.seed(a=args.seed)
obj_attr = import_attr(ig.path_to_tasks, attribute=ig.obj)
f = obj_attr()

X  = np.zeros((0, ig.high_dim))
fX = np.zeros((0, 1))
for i in range(ig.max_evals):
    rand_ind = np.random.uniform(size=(1,ig.high_dim))
    print(rand_ind.shape)
    X  = np.vstack((X, rand_ind             ))
    fX = np.vstack((fX,f.__call__(rand_ind) ))


filename = ig.data_samples_dir + 'pure_random_seed_' + str(args.seed) + '.p'
dict_out = OrderedDict([('Xepisodes', X),('Yepisodes', fX)])
save_dictionary(filename, dict_out)

ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
