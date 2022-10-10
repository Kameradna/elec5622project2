import os
import shutil
import csv
import numpy as np
from tqdm import tqdm

direc = {}
for root in ['data/validation', 'data/training', 'data/test']:
  for file in os.listdir(root):
    print(file.strip('.png'))
    direc[file.strip('.png')] = root
# print(direc)

with open('data/data.csv','r') as f:
  csvReader = csv.reader(f)
  npline = np.array(next(csvReader, None))
  
  for line in tqdm(csvReader):
    uid = line[0].rjust(5,'0')
    if os.path.exists(f"{direc[uid]}/{line[1]}") != True:
      os.mkdir(f"{direc[uid]}/{line[1]}")
    print(f"Moving {direc[uid]}/{uid}.png to {direc[uid]}/{line[1]}/{uid}.png")
    shutil.move(f"{direc[uid]}/{uid}.png",f"{direc[uid]}/{line[1]}/{uid}.png")

  
