import numpy as np
import pandas as pd
from pandas import read_csv
import math
import matplotlib.pyplot as plt
import glob
from copy import deepcopy
import sys
import os

directory = "/eos/user/s/swaldych/smart_pix/preprocess/"
if not os.path.exists(directory):
    os.makedirs(directory)
        
sensor_geom = '50x12P5'
noise_threshold=0
threshold = 0.2
output_file = open(directory+"output_"+sensor_geom+"_"+str(noise_thresholdt)+"NoiseThresh.txt", "w")

qm_charge_levels = [400, 1600, 2400]
qm_quant_values = [0,1,2,3]

train_dataset_name = 'dataset_3s' # for train datasets
test_dataset_name = 'dataset_2s' # for location of test (physical pT) datasets

dataset_savedir = f'/eos/user/s/swaldych/smart_pix/dataset_3s_{noise_threshold}NoiseThresh'
