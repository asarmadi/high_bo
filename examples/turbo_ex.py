import argparse
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))

import numpy as np
from utils.load_save import save_dictionary
from utils.import_modules import import_attr
from utils.name_file import name_turbo
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

turbo_att = import_attr(ig.path_to_turbo, attribute=ig.turbo)

if not os.path.isdir(ig.data_samples_dir):
   os.mkdir(ig.data_samples_dir)

turbo_ = turbo_att(
    f=f,                                     # Handle to objective function
    lb=f.lb,                                 # Numpy array specifying lower bounds
    ub=f.ub,                                 # Numpy array specifying upper bounds
    n_init=ig.n_init,                        # Number of initial bounds from an Symmetric Latin hypercube design
    max_evals=ig.max_evals,                  # Maximum number of evaluations
    n_trust_regions=ig.n_trust_regions,      # Number of trust regions
    batch_size=ig.batch_size,                # How large batch size TuRBO uses
    verbose=ig.verbose,                      # Print information from each batch
    use_ard=ig.use_ard,                      # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=ig.max_cholesky_size,  # When we switch from Cholesky to Lanczos
    n_training_steps=ig.n_training_steps,    # Number of steps of ADAM to learn the hypers
    min_cuda=ig.min_cuda,                    # Run on the CPU for small datasets
    device=ig.device,                        # "cpu" or "cuda"
    dtype=ig.dtype,                          # float64 or float32
    initialization=ig.ea_type,
)

turbo_.optimize(str(args.seed))

X = turbo_.X  # Evaluated points
fX = turbo_.fX  # Observed values

filename = ig.data_samples_dir + name_turbo() + 'seed_' + str(args.seed) + '.p'
dict_out = OrderedDict([('Xepisodes', X),('Yepisodes', fX)])
save_dictionary(filename, dict_out)

ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))
