import random
from rename import metrics, algs
import pandas as pd

for metric in metrics:
    data_frames = []
    for alg in algs:
        df = pd.read_csv(f'D:\\Pyproj\\experiment\\CRMAPPO\\{alg}\\{metric}.csv')
        data_frames.append(
            df[['Step', 'average', 'std_dev']].rename(columns={'average': f'{alg}', 'std_dev': f'{alg}_std'}))

    merged_data = pd.concat(data_frames, axis=1)
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

    merged_data.to_csv(f'D:\\Pyproj\\experiment\\CRMAPPO\\{metric}_all.csv', index=False)
