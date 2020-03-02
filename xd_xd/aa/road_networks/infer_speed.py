from logging import getLogger
from pathlib import Path
import os
import sys
import cv2
import time
import logging
import argparse
import json
import skimage.io
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
import scipy.spatial
from statsmodels.stats.weightstats import DescrStatsW


logger = getLogger('aa')


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """

    weighted_stats = DescrStatsW(values, weights=weights, ddof=0)

    mean = weighted_stats.mean     # weighted mean of data (equivalent to np.average(array, weights=weights))
    std = weighted_stats.std       # standard deviation with default degrees of freedom correction
    var = weighted_stats.var       # variance with default degrees of freedom correction

    return (mean, std, var)


def load_speed_conversion_dict_contin(csv_loc):
    '''Load speed to burn_val conversion dataframe
    and create conversion dictionary.
    Assume continuous conversion'''
    df_ = pd.read_csv(csv_loc, index_col=0)
    # get dict of pixel value to speed
    df_tmp = df_.set_index('burn_val')
    dic = df_tmp.to_dict()['speed']
    return df_, dic


def get_nearest_key(dic, val):
    '''Get nearest dic key to the input val'''
    myList = dic
    key = min(myList, key=lambda x:abs(x-val))
    return key


def load_speed_conversion_dict_binned(csv_loc, speed_increment=5):
    '''Load speed to burn_val conversion dataframe
    and create conversion dictionary.
    speed_increment is the increment of speed limits in mph
    10 mph bins go from 1-10, and 21-30, etc.
    breakdown of speed limits in training set:
        # 15.0 5143
        # 18.75 6232
        # 20.0 18098
        # 22.5 347
        # 25.0 16526
        # 26.25 50
        # 30.0 734
        # 33.75 26
        # 35.0 3583
        # 41.25 16
        # 45.0 2991
        # 48.75 17
        # 55.0 2081
        # 65.0 407
    Assuming a similar distribut in the test set allos us to 
    '''
    
    df_ = pd.read_csv(csv_loc, index_col=0)
    # get dict of channel to speed
    df = df_[['channel', 'speed']]
    
    # simple mean of speed bins
    means = df.groupby(['channel']).mean().astype(int)
    dic = means.to_dict()['speed']   

    # speeds are every 5 mph, so take the mean of the 5 mph bins
    #z = [tmp for tmp in a if tmp%5==0]   
    # or just add increment/2 to means...
    dic.update((x, y+speed_increment/2) for x, y in dic.items())
    
    ########## 
    # OPTIONAL
    # if using 10mph bins, update dic
    dic[0] = 7.5
    dic[1] = 17.5 # 15, 18.75, and 20 are all common
    dic[2] = 25  # 25 mph speed limit is ubiquitous
    dic[3] = 35  # 35 mph speed limit is ubiquitous
    dic[4] = 45  # 45 mph speed limit is ubiquitous
    dic[5] = 55  # 55 mph speed limit is ubiquitous
    dic[6] = 65  # 65 mph speed limit is ubiquitous
    return df_, dic


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


###############################################################################
def get_patch_speed_singlechannel(patch, conv_dict, percentile=80,
                                 verbose=False, super_verbose=False):
    '''
    Get the estiamted speed of the given patch where the value of the 2-D
    mask translates directly to speed'''
    
    # get mean of all high values
    thresh = np.percentile(patch, percentile)
    idxs = np.where(patch >= thresh)
    patch_filt = patch[idxs]
    # get mean of high percentiles
    #pixel_val = np.mean(patch_filt)
    pixel_val = np.median(patch_filt)
    
    # get nearest key to pixel_val
    key = get_nearest_key(conv_dict, pixel_val)
    speed = conv_dict[key]
    
    if verbose:
        logger.info("patch_filt: " + str(patch_filt))
        logger.info("conv_dict: " + str(conv_dict))
        logger.info("key: " + str(key))
        logger.info("speed: " + str(speed))
    

#    ########## 
#    # OPTIONAL
#    # bin to 10mph bins
#    myList = [7.5,17.5, 25, 35, 45, 55, 65]
#    speed = min(myList, key=lambda x:abs(x-speed))
#    ########## 
    
    return speed, patch_filt
   
###############################################################################
def get_patch_speed_multichannel(patch, conv_dict, min_z=128, 
                                 weighted=True, percentile=90,
                                 verbose=False, super_verbose=False):
    '''
    Get the estiamted speed of the given patch where each channel
    corresponds to a different speed bin.  
    Assume patch has shape: (channels, h, w).
    If weighted, take weighted mean of each band above threshold,
    else assign speed to max band'''
    
    # set minimum speed if no channel his min_z
    min_speed = -1
    #min_speed = np.min(list(conv_dict.values()))
    
    # could use mean, max, or percentile
    #z_val_vec = np.rint(np.max(patch, axis=(1,2))).astype(int)
    #z_val_vec = np.rint(np.mean(patch, axis=(1,2))).astype(int)
    z_val_vec = np.rint(np.percentile(patch, percentile, 
                                      axis=(1,2)).astype(int))


    if verbose:
        logger.info("    z_val_vec: " + str(z_val_vec))
        
    if not weighted:
        best_idx = np.argmax(z_val_vec)
        if z_val_vec[best_idx] >= min_z:
            speed_out = conv_dict[best_idx]
        else:
            speed_out = min_speed
            
    else:
        # Take a weighted average of all bands with all values above the threshold
        speeds, weights = [], []
        for band, speed in conv_dict.items():
            if super_verbose:
                logger.info("    band: " + str(band), "speed;", str(speed))
            if z_val_vec[band] > min_z:
                speeds.append(speed)
                weights.append(z_val_vec[band])   
        # get mean speed
        if len(speeds) == 0:
            speed_out = min_speed
        # get weighted speed
        else:
            speed_out, std, var = weighted_avg_and_std(speeds, weights)
            if verbose:
                logger.info("    speeds: " + str(speeds), "weights: " + str(weights))
                logger.info("    w_mean: " + str(speed_out), "std: " + str(std))
            if (type(speed_out) == list) or (type(speed_out) == np.ndarray):
                speed_out = speed_out[0]
            
    #if z_val_vec[4] > 50:
    #    return
    
    if verbose:
        logger.info("    speed_out: " + str(speed_out))
                       
    return speed_out, z_val_vec

###############################################################################
def get_edge_time_properties(mask, edge_data, conv_dict,
                             min_z=128, dx=4, dy=4, percentile=80,
                             use_totband=True, use_weighted_mean=True,
                             variable_edge_speed=False,
                             verbose=False):
    '''
    Get speed estimate from proposal mask and graph edge_data by
    inferring the speed along each segment based on the coordinates in the 
    output mask,
    min_z is the minimum mask value to consider a hit for speed
    dx, dy is the patch size to average for speed
    if totband, the final band of the mask is assumed to just be a binary
        road mask and not correspond to a speed bin
    if weighted_mean, sompeu the weighted mean of speeds in the multichannel
        case
    '''

    meters_to_miles = 0.000621371
    
    if len(mask.shape) > 2:
        multichannel = True
    else:
        multichannel = False

    # get coords
    if verbose:
        logger.info("edge_data: " + str(edge_data))

    length_pix = np.sum([edge_data['length_pix']])
    length_m = edge_data['length']
    pix_to_meters = length_m / length_pix
    length_miles = meters_to_miles * length_m
    if verbose:
        logger.info("length_pix: " + str(length_pix))
        logger.info("length_m: " + str(length_m))
        logger.info("length_miles: " + str(length_miles))
        logger.info("pix_to_meters: " + str(pix_to_meters))

    wkt_pix = edge_data['wkt_pix']
    #geom_pix = shapely.wkt.loads(wkt_pix)
    geom_pix = edge_data['geometry_pix']
    # get points
    coords = list(geom_pix.coords)
    if verbose:
        logger.info("type geom_pix: " + str(type(geom_pix))  )
        logger.info("wkt_pix: " + str(wkt_pix))
        logger.info("geom_pix: " + str(geom_pix))
        logger.info("coords: " + str(coords))

    # get midpoints of each segment in the linestring
    x_mids, y_mids, dls = get_linestring_midpoints(geom_pix)
    if verbose:
        logger.info("x_mids: " + str(x_mids))
        logger.info("y_mids: " + str(y_mids))
        logger.info("dls: " + str(dls))
        logger.info("np.sum dls (pix): " + str(np.sum(dls)))
        logger.info("edge_data.length (m): " + str(edge_data['length']))

    # for each midpoint:
    #   1. access that portion of the mask, +/- desired pixels
    #   2. get speed and travel time
    #   Sum the travel time for each segment to get the total speed, this
    #   means that the speed is variable along the edge

    # could also sample the mask at each point in the linestring (except
    #  endpoits), which would give a denser estimate of speed)
    tot_hours = 0
    speed_arr = []
    z_arr = []
    for j,(x,y, dl_pix) in enumerate(zip(x_mids, y_mids, dls)):
        x0, x1 = max(0, x-dx), x+dx + 1
        y0, y1 = max(0, y-dy), y+dy + 1
        if verbose:
            logger.info("  x, y, dl: " + str(x), str(y), str(dl_pix))

        # multichannel case...
        if multichannel:
            patch = mask[:, y0:y1, x0:x1]
            if use_totband:
                # assume the final channel is total, so cut it out
                nchannels, h, w = mask.shape
                patch = patch[:nchannels-1,:,:]
            if verbose:
                logger.info("  patch.shape: " + str(patch.shape))
            # get estimated speed of mask patch
            speed_mph_seg, z = get_patch_speed_multichannel(patch, conv_dict,
                                 percentile=percentile,
                                 min_z=min_z, weighted=use_weighted_mean,
                                 verbose=verbose)

        else:
            #logger.info("Still need to write the code for single channel continuous masks...")
            patch = mask[y0:y1, x0:x1]
            z = 0
            speed_mph_seg, _ = get_patch_speed_singlechannel(patch, conv_dict,
                                 percentile=percentile,
                                 verbose=verbose, super_verbose=False)

        # add to arrays
        speed_arr.append(speed_mph_seg)
        z_arr.append(z)
        length_m_seg = dl_pix * pix_to_meters
        length_miles_seg = meters_to_miles * length_m_seg
        hours = length_miles_seg / speed_mph_seg
        tot_hours += hours
        if verbose:
            logger.info("  speed_mph_seg: " + str(speed_mph_seg))
            logger.info("  dl_pix: " + str(dl_pix), "length_m_seg", str(length_m_seg),
                   "length_miles_seg: " + str(length_miles_seg))
            logger.info("  hours: " + str(hours))


    # Get edge properties
    if variable_edge_speed:
        mean_speed_mph = length_miles / tot_hours
    else:
        # assume that the edge has a constant speed, so guess the total speed
        if multichannel:
            # get most common channel, assign that channel as mean speed
            z_arr = np.array(z_arr)
            # sum along the channels
            z_vec = np.sum(z_arr, axis=0)
            # get max speed value
            channel_best = np.argmax(z_vec)
            if verbose:
                logger.info("z_arr: " + str(z_arr))
                logger.info("z_vec: " + str(z_vec))
                logger.info("channel_best: " + str(channel_best))
            mean_speed_mph = conv_dict[channel_best]
            # reassign total hours
            tot_hours = length_miles / mean_speed_mph
        else:
            # or always use variable edge speed?
            mean_speed_mph = length_miles / tot_hours

    return tot_hours, mean_speed_mph, length_miles


###############################################################################
def infer_travel_time(G_,
                      mask,
                      conv_dict,
                      min_z=128,
                      dx=4,
                      dy=4,
                      percentile=90,
                      use_totband=True,
                      use_weighted_mean=True,
                      variable_edge_speed=False,
                      verbose=False):

    '''Get an estimate of the average speed and travel time of each edge
    in the graph from the mask and conversion dictionary
    For each edge, get the geometry in pixel coords
      For each point, get the neareast neighbors in the maks and infer
      the local speed'''

    mph_to_mps = 0.44704   # miles per hour to meters per second

    for i,(u, v, edge_data) in enumerate(G_.edges(data=True)):
        if verbose: #(i % 100) == 0:
            logger.info("\n" + str(i) + " " + str(u) + " " + str(v) + " " \
                         + str(edge_data))
        if (i % 1000) == 0:
            logger.info(str(i) + " / " + str(len(G_.edges())) + " edges")

        tot_hours, mean_speed_mph, length_miles = \
                get_edge_time_properties(mask, edge_data, conv_dict,
                             min_z=min_z, dx=dx, dy=dy,
                             percentile=percentile,
                             use_totband=use_totband,
                             use_weighted_mean=use_weighted_mean,
                             variable_edge_speed=variable_edge_speed,
                             verbose=verbose)
        # update edges
        edge_data['Travel Time (h)'] = tot_hours
        edge_data['inferred_speed_mph'] = np.round(mean_speed_mph, 2)
        edge_data['length_miles'] = length_miles
        edge_data['inferred_speed_mps'] = np.round(mean_speed_mph * mph_to_mps, 2)
        edge_data['travel_time_s'] = np.round(3600. * tot_hours, 3)
        # edge_data['travel_time'] = np.round(3600. * tot_hours, 3)

    return G_

###############################################################################
def add_travel_time_dir(graph_dir, mask_dir, conv_dict, graph_dir_out,
                      min_z=128, dx=4, dy=4, percentile=90,
                      use_totband=True, use_weighted_mean=True,
                      variable_edge_speed=False, mask_prefix='',
                      save_shapefiles=True,
                      verbose=False):
    '''Update graph properties to include travel time for entire directory'''
    pickle_protocol = 4     # 4 is most recent, python 2.7 can't read 4

    logger.info("Updating graph properties to include travel time")
    logger.info("  Writing to: " + str(graph_dir_out))
    os.makedirs(graph_dir_out, exist_ok=True)

    image_names = sorted([z for z in os.listdir(mask_dir) if z.endswith('.tif')])
    for i,image_name in enumerate(image_names):
        im_root = image_name.split('.')[0]
        if len(mask_prefix) > 0:
            im_root = im_root.split(mask_prefix)[-1]
        out_file = os.path.join(graph_dir_out, im_root + '.gpickle')

        if (i % 1) == 0:
            logger.info("\n" + str(i+1) + " / " + str(len(image_names)) + " " + image_name + " " + im_root)
        mask_path = os.path.join(mask_dir, image_name)
        graph_path = os.path.join(graph_dir,  im_root + '.gpickle')

        if not os.path.exists(graph_path):
            logger.info("  ", i, "DNE, skipping: " + str(graph_path))
            return
            # continue

        mask = skimage.io.imread(mask_path)
        G_raw = nx.read_gpickle(graph_path)

        # see if it's empty
        if len(G_raw.nodes()) == 0:
            nx.write_gpickle(G_raw, out_file, protocol=pickle_protocol)
            continue

        G = infer_travel_time(G_raw, mask, conv_dict,
                             min_z=min_z, dx=dx, dy=dy,
                             percentile=percentile,
                             use_totband=use_totband,
                             use_weighted_mean=use_weighted_mean,
                             variable_edge_speed=variable_edge_speed,
                             verbose=verbose)
        G = G.to_undirected()
        nx.write_gpickle(G, out_file, protocol=pickle_protocol)
    return


def infer_speed(conf):
    '''See _arr_slicing_speed.ipynb for better tests'''

    t0 = time.time()
    percentile = 85
    dx, dy = 4, 4   # nearest neighbors patch size
    min_z = 128     # min z value to consider a hit

    #width_key, width_mult = 4, 1   # constant width
    if conf.num_classes == 8:
        use_totband = True
    else:
        use_totband = False

    save_shapefiles = False
    use_weighted_mean = True
    variable_edge_speed = False
    verbose = False

    # output pkl
    graph_dir = "{}/working/sp5r2/models/graphs/{}".format(
        "/wdata", conf.modelname)
    Path(graph_dir).mkdir(parents=True, exist_ok=True)

    preds_dirname = conf.modelname.replace('_th06', '')

    merge_dir = (
        "/wdata" + "/working/sp5r2/models/preds/" +
        preds_dirname + "/merged_test")

    mask_dir = merge_dir
    mask_prefix = ''

    if conf.num_folds == 1:
        folds_dir = (
            "/wdata" + "/working/sp5r2/models/preds/" +
            preds_dirname + "/fold0_test")
        mask_dir = folds_dir
        mask_prefix = 'fold0_'

    # output pkl
    graph_speed_dir = "{}/working/sp5r2/models/graphs_speed/{}".format(
        "/wdata", conf.modelname)
    Path(graph_speed_dir).mkdir(parents=True, exist_ok=True)

    logger.info("graph_speed_dir: " + graph_speed_dir)

    # speed conversion dataframes (see _speed_data_prep.ipynb)
    speed_conversion_file_binned = os.path.join(
        "/wdata" + '/input/train/masks_base/',
        'roads_train_speed_conversion_binned.csv',
    )

    # load conversion file
    # get the conversion diction between pixel mask values and road speed (mph)
    assert conf.num_classes > 1
    conv_df, conv_dict \
        = load_speed_conversion_dict_binned(speed_conversion_file_binned)
    logger.info("speed conv_dict: " + str(conv_dict))

    # Add travel time to entire dir
    add_travel_time_dir(graph_dir,
                        mask_dir,
                        conv_dict,
                        graph_speed_dir,
                        min_z=min_z,
                        dx=dx, dy=dy,
                        percentile=percentile,
                        use_totband=use_totband,
                        use_weighted_mean=use_weighted_mean,
                        variable_edge_speed=variable_edge_speed,
                        mask_prefix=mask_prefix,
                        save_shapefiles=save_shapefiles,
                        verbose=verbose)

    t1 = time.time()
    logger.info("Time to execute add_travel_time_dir(): {x} seconds".format(x=t1-t0))
