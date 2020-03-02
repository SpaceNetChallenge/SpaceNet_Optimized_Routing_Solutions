# -*- coding: utf-8 -*-
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
from os import path, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import timeit
import cv2
from tqdm import tqdm

import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool

from shapely.geometry.linestring import LineString
# from skimage.morphology import skeletonize_3d, square, erosion, dilation, medial_axis
# from skimage.measure import label, regionprops, approximate_polygon
from math import hypot, sin, cos, asin, acos, radians
from sklearn.neighbors import KDTree
from shapely.wkt import dumps, loads

import scipy

import utm
#pip install utm
import gdal
gdal.UseExceptions()
import osr
import ogr
#conda install gdal

import ntpath
from shapely.geometry import mapping, Point, LineString

# import matplotlib.pyplot as plt
# import seaborn as sns

pred_folders = ['/wdata/test_pred', '/wdata/test_pred_960']

speed_bins = np.array([15, 18.75, 20, 25, 30, 35, 45, 55, 65])

# test_folders = ['/data/SN5_roads/test_public/AOI_7_Moscow', '/data/SN5_roads/test_public/AOI_8_Mumbai', '/data/SN5_roads/test_public/AOI_9_San_Juan']
test_folders = []
for i in range(1, len(sys.argv) - 1):
    test_folders.append(sys.argv[i])

df = pd.read_csv(path.join('/wdata', 'solution_length.csv'), header=None)
df.columns = ['ImageId', 'WKT_Pix']

# example GDAL error handler function
def gdal_error_handler(err_class, err_num, err_msg):
    errtype = {
            gdal.CE_None:'None',
            gdal.CE_Debug:'Debug',
            gdal.CE_Warning:'Warning',
            gdal.CE_Failure:'Failure',
            gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print('Error Number: ', (err_num))
    print('Error Type: ', (err_class))
    print('Error Message: ', (err_msg))
    
gdal.PushErrorHandler(gdal_error_handler)

# from https://github.com/CosmiQ/cresi
def get_linestring_midpoints(geom):
    '''Get midpoints of each line segment in the line.
    Also return the length of each segment, assuming cartesian coordinates'''
    coords = list(geom.coords)
    N = len(coords)
    x_mids, y_mids, dls = [], [], []
    for i in range(N-1):
        (x0, y0) = coords[i]
        (x1, y1) = coords[i+1]
        x_mids.append(np.rint(0.5 * (x0 + x1)))
        y_mids.append(np.rint(0.5 * (y0 + y1)))
        dl = scipy.spatial.distance.euclidean(coords[i], coords[i+1])
        dls. append(dl)
    return np.array(x_mids).astype(int), np.array(y_mids).astype(int), \
                np.array(dls)


def pixelToGeoCoord(xPix, yPix, inputRaster, sourceSR='', geomTransform='', targetSR=''):
    '''from spacenet geotools'''

    if targetSR =='':
        performReprojection=False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection=True

    if geomTransform=='':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())

    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

    xCoord = (xPix * pixelWidth) + xOrigin
    yCoord = (yPix * pixelHeight) + yOrigin
    geom.AddPoint(xCoord, yCoord)

    if performReprojection:
        if sourceSR=='':
            srcRaster = gdal.Open(inputRaster)
            sourceSR = osr.SpatialReference()
            sourceSR.ImportFromWkt(srcRaster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(sourceSR, targetSR)
        geom.Transform(coord_trans)

    return (geom.GetX(), geom.GetY())


def convert_pix_lstring_to_geo(wkt_lstring, im_file, 
                               utm_zone=None, utm_letter=None, verbose=False):
    '''Convert linestring in pixel coords to geo coords
    If zone or letter changes inthe middle of line, it's all screwed up, so
    force zone and letter based on first point
    (latitude, longitude, force_zone_number=None, force_zone_letter=None)
    Or just force utm zone and letter explicitly
        '''
    shape = wkt_lstring  #shapely.wkt.loads(lstring)
    x_pixs, y_pixs = shape.coords.xy
    coords_latlon = []
    coords_utm = []
    for i,(x,y) in enumerate(zip(x_pixs, y_pixs)):
        
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
        lon, lat = pixelToGeoCoord(x, y, im_file, targetSR=targetSR)

        if utm_zone and utm_letter:
            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon,
                force_zone_number=utm_zone, force_zone_letter=utm_letter)
        else:
            [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)
        
        if verbose:
            print("lat lon, utm_east, utm_north, utm_zone, utm_letter]",
                [lat, lon, utm_east, utm_north, utm_zone, utm_letter])
        coords_utm.append([utm_east, utm_north])
        coords_latlon.append([lon, lat])
    
    lstring_latlon = LineString([Point(z) for z in coords_latlon])
    lstring_utm = LineString([Point(z) for z in coords_utm])
    
    return lstring_latlon, lstring_utm, utm_zone, utm_letter

meters_to_miles = 0.000621371

###########


def get_linestring_keypoints(geom):
    coords = list(geom.coords)
    N = len(coords)
    xs, ys, dls = [], [], []
    for i in range(N-1):
        xs.append([])
        ys.append([])

        (x0, y0) = coords[i]
        (x1, y1) = coords[i+1]
        
        xs[i].append(0.5 * x0 + 0.5 * x1)
        ys[i].append(0.5 * y0 + 0.5 * y1)
        xs[i].append(0.75 * x0 + 0.25 * x1)
        ys[i].append(0.75 * y0 + 0.25 * y1)
        xs[i].append(0.25 * x0 + 0.75 * x1)
        ys[i].append(0.25 * y0 + 0.75 * y1)
        xs[i].append(0.9 * x0 + 0.1 * x1)
        ys[i].append(0.9 * y0 + 0.1 * y1)
        xs[i].append(0.1 * x0 + 0.9 * x1)
        ys[i].append(0.1 * y0 + 0.9 * y1)
        xs[i].append(0.35 * x0 + 0.65 * x1)
        ys[i].append(0.35 * y0 + 0.65 * y1)
        xs[i].append(0.65 * x0 + 0.35 * x1)
        ys[i].append(0.65 * y0 + 0.35 * y1)

        dl = scipy.spatial.distance.euclidean(coords[i], coords[i+1])
        dls. append(dl)
    return xs, ys, np.asarray(dls)



def process_file(fn):
    img_id = ntpath.basename(fn)[0:-4]
    img_id = img_id.replace('_PS-MS', '')
    
    im_file = fn
    
    msks = []
    for pred_folder in pred_folders:
        msk0 = cv2.imread(path.join(pred_folder, img_id + '_speed0.png'), cv2.IMREAD_UNCHANGED)
        msk1 = cv2.imread(path.join(pred_folder, img_id + '_speed1.png'), cv2.IMREAD_UNCHANGED)
        msk2 = cv2.imread(path.join(pred_folder, img_id + '_speed2.png'), cv2.IMREAD_UNCHANGED)
        msk = np.concatenate((msk0, msk1, msk2), axis=2)
        if msk.shape[0] < 1306:
            msk = cv2.resize(msk, (1300, 1300))
            msk = np.pad(msk, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        msks.append(msk)
    msks = np.asarray(msks)
    msk = msks.mean(axis=0)
    msk = msk[6:1306, 6:1306].astype('uint8')
    
    vals = df[(df['ImageId'] == img_id)]['WKT_Pix'].values
    res_rows = []
    for v in vals:
        if v == 'LINESTRING EMPTY':
            return [{'ImageId': img_id, 'WKT_Pix': 'LINESTRING EMPTY', 'length_m': 0, 'travel_time_s': 0}]
        
        l = loads(v)
        
        lstring_latlon, lstring_utm, utm_zone, utm_letter = convert_pix_lstring_to_geo(l, im_file)
    
        length = lstring_utm.length
        length_miles = length * meters_to_miles
        
#         x_mids, y_mids, dls = get_linestring_midpoints(l)
        xs, ys, dls = get_linestring_keypoints(l)
        
        _sz = 4

#         speed = []
#         if x_mids.shape[0] > 0:
#             for i in range(x_mids.shape[0]):
#                 x0 = max(0, x_mids[i] - _sz)
#                 x1= min(1300, x_mids[i] + _sz)
#                 y0 = max(0, y_mids[i] - _sz)
#                 y1= min(1300, y_mids[i] + _sz)
#                 patch = msk[y0:y1, x0:x1]
#                 means = patch.mean(axis=(0, 1))
                
#                 if means.sum() == 0:
#                     speed.append(25)
#                 else:
#                     means /= means.sum()
#                     _s = (speed_bins * means).sum()
#                     if _s < 15:
#                         _s = 15
#                     if _s > 65:
#                         _s = 65
#                     speed.append(_s)
                    
        speed = []
        if len(xs) > 0:
            for i in range(len(xs)):
                seg_speeds = []
                for j in range(len(xs[i])):
                    x0 = max(0, int(xs[i][j] - _sz))
                    x1= min(1300, int(xs[i][j] + _sz))
                    y0 = max(0, int(ys[i][j] - _sz))
                    y1= min(1300, int(ys[i][j] + _sz))
                    
                    patch = msk[y0:y1, x0:x1]
                    means = patch.mean(axis=(0, 1))

                if means.sum() == 0:
                    seg_speeds.append(25)
                else:
                    means /= means.sum()
                    _s = (speed_bins * means).sum()
                    if _s < 15:
                        _s = 15
                    if _s > 65:
                        _s = 65
                    seg_speeds.append(_s)
                speed.append(np.mean(seg_speeds))
                    
        speed = np.asarray(speed)
        dls /= dls.sum()
        speed = (speed * dls).sum()
        
        if speed < 15:
            speed = 15
        if speed > 65:
            speed = 65
                        
        hours = length_miles / speed
        
        travel_time_s = np.round(3600. * hours, 3)
        
        res_rows.append({'ImageId': img_id, 'WKT_Pix': v, 'length_m': length, 'travel_time_s': travel_time_s})


    return res_rows



if __name__ == '__main__':
    t0 = timeit.default_timer()
           
    out_file = sys.argv[-1]
        
    # out_file = '/wdata/solution.csv'
        
    all_files = []
    for d in test_folders:
        for f in listdir(path.join(d, 'PS-MS')):
            if '.tif' in f:
                all_files.append(path.join(d, 'PS-MS', f))
        
#     for fn in tqdm(all_files):
#         process_file(fn)
        
    with Pool() as pool:
        results = pool.map(process_file, all_files)
    
    res_rows = []
    for i in range(len(results)):
        res_rows.extend(results[i])
        
    sub = pd.DataFrame(res_rows, columns=['ImageId', 'WKT_Pix', 'length_m', 'travel_time_s'])
    sub.to_csv(path.join('/wdata', out_file), index=False, header=False)
                
    elapsed = timeit.default_timer() - t0
    print('Submission file created! Time: {:.3f} min'.format(elapsed / 60))