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

# from matplotlib import pyplot as plt
# import seaborn as sns

cities = []
for i in range(1, len(sys.argv)):
    d = sys.argv[i]
    city = d.split('//')[-1].split('AOI_')[1].split('_')
    city = 'AOI_' + city[0] + '_' + city[1]

    csv_file = None
    for f in listdir(d):
        if f.endswith('_simp.csv'):
            csv_file = f
            break
    print(city)
    cities.append((city, d, csv_file))

cities_idxs = {}
for i in range(len(cities)):
    cities_idxs[cities[i][0]] = i


dfs = []
for i in range(len(cities)):
    df = pd.read_csv(path.join(cities[i][1], cities[i][2]))
    df['speed'] = (df['length_m'] / 1609.344) / (df['travel_time_s'] / 60 / 60)
    df['speed'] = df['speed'].fillna(15).round(4)
    dfs.append(df)
    
train_png_full = '/wdata/train_png'
train_png2_full = '/wdata/train_png_5_3_0'
train_png3_full = '/wdata/train_png_pan_6_7'

train_png = '/wdata/train_png_960'
train_png2 = '/wdata/train_png_5_3_0_960'
train_png3 = '/wdata/train_png_pan_6_7_960'

masks_dir_full = '/wdata/masks'
masks_dir = '/wdata/masks_960'

speed_bins = [15, 18.75, 20, 25, 30, 35, 45, 55, 65]
# feature for speed - touch border or not
thickness = 12
radius = 16
ratio = 1
def process_image(img_id):
    img_id0 = img_id
    
    _sep = '_img'
    if '_chip' in img_id:
        _sep = '_chip'
        
    tmp = img_id.split(_sep)
    city = tmp[0].split('train_')[1]
    
    img_bgr = cv2.imread(path.join(train_png_full, img_id + '.png'), cv2.IMREAD_UNCHANGED)
    img_bgr = cv2.resize(img_bgr, (960, 960))
    cv2.imwrite(path.join(train_png, img_id + '.png'), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    img_0_3_5 = cv2.imread(path.join(train_png2_full, img_id + '.png'), cv2.IMREAD_UNCHANGED)
    img_0_3_5 = cv2.resize(img_0_3_5, (960, 960))
    cv2.imwrite(path.join(train_png2, img_id + '.png'), img_0_3_5, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    img_pan_6_7 = cv2.imread(path.join(train_png3_full, img_id + '.png'), cv2.IMREAD_UNCHANGED)
    img_pan_6_7 = cv2.resize(img_pan_6_7, (960, 960))
    cv2.imwrite(path.join(train_png3, img_id + '.png'), img_pan_6_7, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    msk = cv2.imread(path.join(masks_dir_full, img_id + '.png'), cv2.IMREAD_UNCHANGED)
    msk = cv2.resize(msk, (960, 960))
    cv2.imwrite(path.join(masks_dir, img_id + '.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    msk = cv2.imread(path.join(masks_dir_full, img_id + '_speed0.png'), cv2.IMREAD_UNCHANGED)
    msk = cv2.resize(msk, (960, 960))
    cv2.imwrite(path.join(masks_dir, img_id + '_speed0.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    msk = cv2.imread(path.join(masks_dir_full, img_id + '_speed1.png'), cv2.IMREAD_UNCHANGED)
    msk = cv2.resize(msk, (960, 960))
    cv2.imwrite(path.join(masks_dir, img_id + '_speed1.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    msk = cv2.imread(path.join(masks_dir_full, img_id + '_speed2.png'), cv2.IMREAD_UNCHANGED)
    msk = cv2.resize(msk, (960, 960))
    cv2.imwrite(path.join(masks_dir, img_id + '_speed2.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    msk = cv2.imread(path.join(masks_dir_full, img_id + '_speed_cont.png'), cv2.IMREAD_UNCHANGED)
    msk = cv2.resize(msk, (960, 960))
    cv2.imwrite(path.join(masks_dir, img_id + '_speed_cont.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    

if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    makedirs(masks_dir, exist_ok=True)
    makedirs(train_png, exist_ok=True)
    makedirs(train_png2, exist_ok=True)
    makedirs(train_png3, exist_ok=True)
    
    all_ids0 = []
    for df in dfs:
        all_ids0 += df['ImageId'].unique().tolist()

    # for img_id in tqdm(all_ids0):
    #     process_image(img_id)
        
    with Pool() as pool:
        _ = pool.map(process_image, all_ids0)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))