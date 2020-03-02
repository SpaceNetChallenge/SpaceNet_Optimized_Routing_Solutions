import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import cv2
import timeit
from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool
from skimage.morphology import square, dilation, watershed, erosion
from skimage import io
from shapely.wkt import loads

from tqdm import tqdm
import ntpath

# from matplotlib import pyplot as plt
# import seaborn as sns

# test_folders = ['/data/SN5_roads/test_public/AOI_7_Moscow', '/data/SN5_roads/test_public/AOI_8_Mumbai', '/data/SN5_roads/test_public/AOI_9_San_Juan']
test_folders = []
for i in range(1, len(sys.argv) - 1):
    test_folders.append(sys.argv[i])
print("test_folders:", test_folders)

test_png = '/wdata/test_png'
test_png2 = '/wdata/test_png_5_3_0'
test_png3 = '/wdata/test_png_pan_6_7'

test_png_960 = '/wdata/test_png_960'
test_png2_960 = '/wdata/test_png_5_3_0_960'
test_png3_960 = '/wdata/test_png_pan_6_7_960'

def process_image(fn):
    img_id = bn = ntpath.basename(fn)[0:-4]
    img_id = img_id.replace('_PS-MS', '')
    
    img = io.imread(fn)
    
    img_bgr = (np.clip(img[..., [1, 2, 4]], None, 2000) / (2000 / 255)).astype('uint8')
    cv2.imwrite(path.join(test_png, img_id + '.png'), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(test_png_960, img_id + '.png'), cv2.resize(img_bgr, (960, 960)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    img_0_3_5 = (np.clip(img[..., [0, 3, 5]], None, 2000) / (2000 / 255)).astype('uint8')
    cv2.imwrite(path.join(test_png2, img_id + '.png'), img_0_3_5, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(test_png2_960, img_id + '.png'), cv2.resize(img_0_3_5, (960, 960)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    pan = io.imread(fn.replace('_PS-MS_', '_PAN_').replace('PS-MS', 'PAN'))
    pan = pan[..., np.newaxis]
    
    img_pan_6_7 = np.concatenate([pan, img[..., 7:], img[..., 6:7]], axis=2)
    img_pan_6_7 = (np.clip(img_pan_6_7, None, (10000, 2000, 2000)) / (np.array([10000, 2000, 2000]) / 255)).astype('uint8')
    cv2.imwrite(path.join(test_png3, img_id + '.png'), img_pan_6_7, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(test_png3_960, img_id + '.png'), cv2.resize(img_pan_6_7, (960, 960)), [cv2.IMWRITE_PNG_COMPRESSION, 9])
       
    

if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(test_png, exist_ok=True)
    makedirs(test_png2, exist_ok=True)
    makedirs(test_png3, exist_ok=True)
    makedirs(test_png_960, exist_ok=True)
    makedirs(test_png2_960, exist_ok=True)
    makedirs(test_png3_960, exist_ok=True)

    all_files = []
    for d in test_folders:
        for f in listdir(path.join(d, 'PS-MS')):
            if '.tif' in f:
                all_files.append(path.join(d, 'PS-MS', f))
        
    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
