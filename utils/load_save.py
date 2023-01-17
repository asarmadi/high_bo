import numpy as np
import glob
import os
import pickle
from collections import OrderedDict
from utils.import_modules import import_attr

def savefile(filename, array):
    try:
        np.save(filename, array)
    except:
        np.save(filename.replace('home', 'homes'), array)

def loadfile(filename):
    try:
        array = np.load(filename)
    except:
        array = np.load(filename.replace('home', 'homes'))
    return array

def load_dictionary(filename):
    try:
        dict_load = pickle.load(open(filename, 'rb'))
    except:
        dict_load = pickle.load(open(filename.replace('home', 'homes'), 'rb'))
    return dict_load

def save_dictionary(filename, dictionary):
    try:
        pickle.dump(dictionary, open(filename, 'wb'))
    except:
        pickle.dump(dictionary, open(filename.replace('home', 'homes'), 'wb'))

