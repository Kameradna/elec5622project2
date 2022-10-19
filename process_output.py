import os
import pandas as pd
import numpy as np

stats = {
    'base_lr':[],
    # 'lr_step_size':[],
    # 'lr_gamma':[],
    'batch_size':[],
    'test_acc':[],
    'stop_reason':[],
    'step':[],
    'special_sauce':[]
}

for root, dir, files in os.walk('save'):
    for file in files: #some quirks with the saving of _best files, maybe pth has some issues too
        if '.pth' in file or '_best' in file or '.ods' in file:
            print(f"skipping {file}")
            continue
        print(file)
        fileinfo = file.split('_')
        stats['base_lr'].append(float(fileinfo[1].strip('baselr')))
        # stats['lr_step_size'].append(int(fileinfo[2].strip('lrstep')))
        # stats['lr_gamma'].append(float(fileinfo[3].strip('lrgam')))
        stats['batch_size'].append(int(fileinfo[4].strip('bat')))
        stats['test_acc'].append(float(fileinfo[7].strip('.csv')))
        stats['stop_reason'].append(str(fileinfo[6].split('@')[0]))
        stats['step'].append(int(fileinfo[6].split('@')[1]))
        try:
            stats['special_sauce'].append(str(fileinfo[8:12]).strip('.csv'))
        except:
            print('"Forget about it!"')

print(stats)
stats_df = pd.DataFrame(stats)
stats_df.to_csv('grid_search_results.csv')
