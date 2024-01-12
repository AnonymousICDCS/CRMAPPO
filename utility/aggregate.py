import random
from rename import metrics, algs
import pandas as pd

for alg in algs:
    for metric in metrics:
        files = [pd.read_csv(r'D:\Pyproj\experiment\CRMAPPO\{}\{}\seed{}.csv'.format(alg, metric, seed)) for seed in range(5)]
        columns = ['Step'] + [f'seed{seed}' for seed in range(5)]
        data = pd.concat(
            [file[['Step', 'Value']].rename(columns={'Value': f'seed{seed}'}) for file, seed in zip(files, range(5))],axis=1)
        data.columns = columns
        data.to_csv(r'D:\Pyproj\experiment\CRMAPPO\{}\{}.csv'.format(alg, metric), index=False)
