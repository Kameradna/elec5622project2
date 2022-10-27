import os
import pandas as pd
import numpy as np

stats = {
    'extractor':[],
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
        stats['extractor'].append(str(fileinfo[0]))
        stats['base_lr'].append(float(fileinfo[1].strip('baselr')))
        # stats['lr_step_size'].append(int(fileinfo[2].strip('lrstep')))
        # stats['lr_gamma'].append(float(fileinfo[3].strip('lrgam')))
        stats['batch_size'].append(int(fileinfo[4].strip('bat')))
        try: #old way, no spice on the end
            stats['test_acc'].append(float(fileinfo[7].strip('.csv')))
        except: #new way, tolerant to additional details
            stats['test_acc'].append(float(fileinfo[7].split('@')[1]))

        stats['stop_reason'].append(str(fileinfo[6].split('@')[0]))
        stats['step'].append(int(fileinfo[6].split('@')[1]))
        try:
            stats['special_sauce'].append(str(fileinfo[8:]).strip('.csv'))
        except:
            print('"Forget about it!"')

print(stats)
stats_df = pd.DataFrame(stats)
stats_df.to_csv('grid_search_results.csv')
