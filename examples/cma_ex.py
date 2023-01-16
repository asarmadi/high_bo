import cma
import random
import argparse
import sys, os
sys.path.insert(0, os.path.join(sys.path[0], '..'))

from tfbo.utils.import_modules import import_attr
from examples.config_file import config
from tfbo.utils.load_save import save_dictionary
from collections import OrderedDict


parser = argparse.ArgumentParser(description='Testing Weights')
parser.add_argument('--seed', type=int, default = 123, help='Seed number')
args = parser.parse_args()
ig = config()

random.seed(a=args.seed)
obj_attr = import_attr(ig.path_to_tasks, attribute=ig.obj)
f = obj_attr()

x0 = ig.high_dim * [1]  # initial solution
sigma0 = 1    # initial standard deviation to sample new solutions

es = cma.CMAEvolutionStrategy(x0, sigma0, {'bounds': [0, 1]})

i = 0
filename = ig.data_samples_dir + 'cma_seed_' + str(args.seed)  +'.p'
while not es.stop():
    X = es.ask()  # sample len(X) candidate solutions
    es.tell(X, [f.__call__(x) for x in X])
#    cfun.update(es)
    es.logger.add()  # for later plotting
    if i % 100 == 0:
       res_obj = es.result
       dict_out = OrderedDict([('Xepisodes', res_obj.xbest),('Yepisodes', res_obj.fbest)])
       save_dictionary(filename, dict_out)
    i += 1

res_obj = es.result

print(res_obj.xbest, res_obj.fbest)
print(i)
input('enter')

dict_out = OrderedDict([('Xepisodes', res_obj.xbest),('Yepisodes', res_obj.fbest)])
filename = ig.data_samples_dir + 'cma_seed_' + str(args.seed)  +'.p'
save_dictionary(filename, dict_out)
