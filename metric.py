import os
import numpy as np
from os.path import isfile, join
from os import listdir
import glob
from collections import defaultdict

PATH = '/rscratch/data/semantickitti/dataset/sequences'
nums = 22

means_x = defaultdict(list)
means_y = defaultdict(list)
means_z = defaultdict(list)
means_i = defaultdict(list)
means = [means_x, means_y, means_z, means_i]

stds_x = defaultdict(list)
stds_y = defaultdict(list)
stds_z = defaultdict(list)
stds_i = defaultdict(list)
stds = [stds_x, stds_y, stds_z, stds_i]

for i in range(nums):
    fold_num = '0'+str(i) if len(str(i)) == 1 else str(i)
    temp_path = os.path.join(PATH, fold_num, 'velodyne/*.bin')
    dirs = glob.glob(temp_path)
    for d in dirs:
        temp_data = np.fromfile(d, dtype=np.float32).reshape(-1, 4)
        means_xyz = np.mean(temp_data, axis = 0)
        stds_xyz = np.std(temp_data, axis = 0)
        for j in range(4):
            means[j][i].append(means_xyz[j])
            stds[j][i].append(stds_xyz[j])
        #print(np.mean(means[j][i]), stds)
        #exit()
        

# Get Average of all files


for i in range(4):
    all_mean = [] 
    all_std = []
    for k in means[0].keys():
        ans = [np.mean(means[i][k]), np.mean(stds[i][k])]
        print(ans)
        all_mean.append(ans[0])
        all_std.append(ans[1])
    print('For ', i, ' all mean is ', np.mean(all_mean), ' all std is ', np.mean(all_std))







