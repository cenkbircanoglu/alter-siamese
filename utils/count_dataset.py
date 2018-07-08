#!/usr/bin/env python3

import argparse
import os
import glob
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inPlaceDir', type=str,
                        help="Directory to prune in-place.")
    args = parser.parse_args()

    for in_place_dir in glob.glob('%s/**/**' %args.inPlaceDir):
        print(in_place_dir)
        exts = ["jpg", "png"]
        total_num = 0
        for subdir, dirs, files in os.walk(in_place_dir):
            if subdir == args.inPlaceDir:
                continue
            nImgs = 0
            for fName in files:
                (imageClass, imageName) = (os.path.basename(subdir), fName)
                if any(imageName.lower().endswith("." + ext) for ext in exts):
                    nImgs += 1
            total_num += nImgs
        print(total_num)
        with open('results/counts.txt', mode='a') as f:
            dataset = in_place_dir.split('/')[-2]
            data_part = in_place_dir.split('/')[-1]
            f.write('%s %s %s res\n' % (dataset, data_part, str(total_num)))
