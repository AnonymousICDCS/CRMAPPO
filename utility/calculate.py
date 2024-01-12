from rename import metrics, algs
import pandas as pd

for alg in algs:
    for metric in metrics:
        # Read the Excel file
        df = pd.read_csv(r'D:\Pyproj\experiment\CRMAPPO\{}\{}.csv'.format(alg, metric))

        # Specify the columns for which you want to calculate average and standard deviation
        seed_columns = [f'seed{i}' for i in range(5)]

        # Calculate average and standard deviation
        df['average'] = df[seed_columns].mean(axis=1)
        df['std_dev'] = df[seed_columns].std(axis=1)

        # Save the modified DataFrame back to Excel
        df.to_csv(r'D:\Pyproj\experiment\CRMAPPO\{}\{}.csv'.format(alg, metric), index=False)
