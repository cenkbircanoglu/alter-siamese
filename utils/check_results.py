import glob
import os
for losses in glob.glob('/media/cenk/2TB2/alter_siamese/results/**/**/**/'):
    if len(os.listdir(losses)) < 4:
        print(losses)
