from importlib import import_module
import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..'))


def import_attr(path_to_module, attribute):
    if attribute == 'a1' or attribute == 'b1':
       attribute = 'unitree_cost'
    module = import_module(path_to_module.replace('/', '.'))
    module_attr = getattr(module, attribute)
    return module_attr
