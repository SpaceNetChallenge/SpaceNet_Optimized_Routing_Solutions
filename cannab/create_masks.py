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
# print("sys.argv:", sys.argv)

cities = []
for i in range(1, len(sys.argv)):
    d = sys.argv[i]
    print(d)
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
    
train_png = '/wdata/train_png'
train_png2 = '/wdata/train_png_5_3_0'
train_png3 = '/wdata/train_png_pan_6_7'

masks_dir = '/wdata/masks'

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
    
    fn = tmp[0] + '_PS-MS' + _sep + img_id.split(_sep)[1] + '.tif'
    
    img = io.imread(path.join(cities[cities_idxs[city]][1], 'PS-MS', fn))
    
    img_bgr = (np.clip(img[..., [1, 2, 4]], None, 2000) / (2000 / 255)).astype('uint8')
    cv2.imwrite(path.join(train_png, img_id + '.png'), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    img_0_3_5 = (np.clip(img[..., [0, 3, 5]], None, 2000) / (2000 / 255)).astype('uint8')
    cv2.imwrite(path.join(train_png2, img_id + '.png'), img_0_3_5, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    pan = io.imread(path.join(cities[cities_idxs[city]][1], 'PAN', fn.replace('_PS-MS_', '_PAN_')))
    pan = pan[..., np.newaxis]
    
    img_pan_6_7 = np.concatenate([pan, img[..., 7:], img[..., 6:7]], axis=2)
    img_pan_6_7 = (np.clip(img_pan_6_7, None, (10000, 2000, 2000)) / (np.array([10000, 2000, 2000]) / 255)).astype('uint8')
    cv2.imwrite(path.join(train_png3, img_id + '.png'), img_pan_6_7, [cv2.IMWRITE_PNG_COMPRESSION, 9])    
    
    df = dfs[cities_idxs[city]]
    
    vals = df[(df['ImageId'] == img_id0)][['WKT_Pix', 'length_m', 'travel_time_s', 'speed']].values
    
    msk0 = np.zeros((1300, 1300), dtype='uint8')
    msk1 = np.zeros((1300, 1300), dtype='uint8')
    msk2 = np.zeros((1300, 1300), dtype='uint8')
    
    msk_speed = np.zeros((1300, 1300, len(speed_bins)), dtype='uint8')
    msk_speed_cont = np.zeros((1300, 1300), dtype='uint8')
    
    d = {}
    
    for l_id in range(len(vals)):
        l = loads(vals[l_id][0])
        
        if len(l.coords) == 0:
            continue
            
        _s = vals[l_id][3]
        _s_i = -1
        _min_d =  1000
        for _i in range(len(speed_bins)):
            if abs(speed_bins[_i] - _s) < _min_d:
                _min_d = abs(speed_bins[_i] - _s)
                _s_i = _i
        
            
        x, y = l.coords.xy
        for i in range(len(x)):
            x[i] /= ratio
            y[i] /= ratio
        
        x_int = int(round(x[0] * 10))
        y_int = int(round(y[0] * 10))
        h = x_int * 100000 + y_int
        if not (h in d.keys()):
            d[h] = 0
        d[h] = d[h] + 1
        
        for i in range(len(x) - 1):
            x_int = int(round(x[i+1] * 10))
            y_int = int(round(y[i+1] * 10))
            h = x_int * 100000 + y_int
            if not (h in d.keys()):
                d[h] = 0
            if i == len(x) - 2:
                d[h] = d[h] + 1
            else:
                d[h] = d[h] + 2
            cv2.line(msk0, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), 255, thickness)
            _tmp = msk_speed[..., _s_i].copy()
            cv2.line(_tmp, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), 255, thickness)
            msk_speed[..., _s_i] = _tmp
            cv2.line(msk_speed_cont, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), int(_s / 65 * 255), thickness)
    for h in d.keys():
        if d[h] > 2:
            x_int = int(h / 100000)
            y_int = h - x_int * 100000
            x_int = int(x_int / 10)
            y_int = int(y_int / 10)
            cv2.circle(msk1, (x_int, y_int), radius, 255, -1)
    
    msk1 = (msk0 > 0) * msk1
    msk0 = msk0[..., np.newaxis]
    msk1 = msk1[..., np.newaxis]
    msk2 = msk2[..., np.newaxis]
    msk = np.concatenate([msk0, msk1, msk2], axis=2)
    
    for i in range(len(speed_bins) - 1):
        for j in range(i + 1, len(speed_bins)):
            msk_speed[msk_speed[..., len(speed_bins)-i-1] > 127, len(speed_bins)-j-1] = 0

    cv2.imwrite(path.join(masks_dir, img_id + '.png'), msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(masks_dir, img_id + '_speed0.png'), msk_speed[..., :3], [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(masks_dir, img_id + '_speed1.png'), msk_speed[..., 3:6], [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(masks_dir, img_id + '_speed2.png'), msk_speed[..., 6:], [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(masks_dir, img_id + '_speed_cont.png'), msk_speed_cont, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    

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
