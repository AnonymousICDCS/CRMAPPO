import random
import os
import pandas as pd
#alg = 'iteRL'
algs = ['beta=0.0', 'beta=0.2', 'beta=0.4', 'beta=0.6', 'beta=0.8']
metrics = ['reward', 'ave_fps', 'ave_consecutive', 'ave_copying', 'resol_score', 'resol_change']
for alg in algs:
    for metric in metrics:
        path = r'D:\Pyproj\experiment\CRMAPPO\{}\{}'.format(alg, metric)
        file_names = os.listdir(path)
        for i in range(5):
            new_name = 'seed{}.csv'.format(i)
            os.rename(os.path.join(path, file_names[i]), os.path.join(path, new_name))
