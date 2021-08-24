import os
import numpy as np
import shutil
import random
#import torch
root_dir = './images_EndoCV2021/'
train_path='train_endoCV2021.txt'
val_path='val_endoCV2021.txt'

val_ratio = 0.2


# fix seed
random_seed= 42
random.seed(random_seed)
np.random.seed(random_seed)
#torch.manual_seed(random_seed)
#torch.cuda.manual_seed(random_seed)
#torch.backends.cudnn.deterministic = True

#
allFileNames = os.listdir(root_dir)
train_FileNames, val_FileNames = np.split(np.array(allFileNames),
                                                      [int(len(allFileNames)* (1 - val_ratio))])
print(train_FileNames)

with open(train_path, 'w') as train_file:
    for line in train_FileNames:
        line = line.split('.')[0]
        train_file.write(line)
        train_file.write('\n')
with open(val_path, 'w') as val_file:
    for line in val_FileNames:
        line = line.split('.')[0]
        val_file.write(line)
        val_file.write('\n')
