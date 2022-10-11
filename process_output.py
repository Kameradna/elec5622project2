import os
import pandas as pd
import numpy as np

stats = {
    'base_lr':[],
    'lr_step_size':[],
    'lr_gamma':[],
    'batch_size':[],
    'test_acc':[]
}



for root, dir, files in os.walk('save'):
    for file in files:
        fileinfo = file.split('_')
        stats['base_lr'].append(float(fileinfo[1].strip('baselr')))
        stats['lr_step_size'].append(int(fileinfo[2].strip('lrstep')))
        stats['lr_gamma'].append(float(fileinfo[3].strip('lrgam')))
        stats['batch_size'].append(int(fileinfo[4].strip('bat')))
        stats['test_acc'].append(float(fileinfo[7].strip('.csv')))

print(stats)
stats_df = pd.DataFrame(stats)
stats_df.to_csv('grid_search_results.csv')
