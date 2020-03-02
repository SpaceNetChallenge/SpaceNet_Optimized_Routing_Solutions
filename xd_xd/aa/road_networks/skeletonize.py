# -*- coding: utf-8 -*-
from logging import getLogger
from pathlib import Path
import os
import json
import time
import random
import argparse
import logging
from itertools import tee
from collections import OrderedDict #, defaultdict
from multiprocessing import Pool, cpu_count

from scipy.spatial.distance import (
    pdist,
    squareform,
)
from skimage.morphology import (
    skeletonize,
    remove_small_objects,
    remove_small_holes,
)
from matplotlib.pylab import plt
import numpy as np
import pandas as pd
import networkx as nx
import tqdm
import skimage.io
import cv2

from aa.cresi.other_tools import sknw


logger = getLogger('aa')

linestring = "LINESTRING {}"


def clean_sub_graphs(G_, min_length=150, max_nodes_to_skip=100,
                     weight='length_pix', verbose=True,
                     super_verbose=False):
    '''Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length
       (this step great improves processing time)'''

    if len(G_.nodes()) == 0:
        return G_

    # print ("Running clean_sub_graphs...")
    sub_graphs = list(nx.connected_component_subgraphs(G_))
    bad_nodes = []

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue
        else:
            all_lengths = dict(nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            # get all lenghts
            lens = []
            #for u,v in all_lengths.iteritems():
            for u in all_lengths.keys():
                v = all_lengths[u]
                #for uprime, vprime in v.iteritems():
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)
            max_len = np.max(lens)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)

    return G_


# From road_raster.py
###############################################################################
def dl_post_process_pred(mask, glob_thresh=80, kernel_size=9,
                         min_area=2000, contour_smoothing=0.001,
                         adapt_kernel=85, adapt_const=-3,
                         outplot_file='', dpi=500, use_glob_thresh=False,
                         kernel_open=19, verbose=False):
    '''Refine mask file and return both refined mask and skeleton'''

    t0 = time.time()
    kernel_blur = kernel_size #9
    kernel_close = kernel_size #9
    #kernel_open = kernel_size #9

    kernel_close = np.ones((kernel_close,kernel_close), np.uint8)
    kernel_open = np.ones((kernel_open, kernel_open), np.uint8)

    blur = cv2.medianBlur(mask, kernel_blur)

    # global thresh
    glob_thresh_arr = cv2.threshold(blur, glob_thresh, 1, cv2.THRESH_BINARY)[1]
    glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)

    t1 = time.time()
    # print ("Time to compute open(), close(), and get thresholds:", t1-t0, "seconds")

    if use_glob_thresh:
        mask_thresh = glob_thresh_arr_smooth
    else:
        adapt_thresh = cv2.adaptiveThreshold(mask,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,adapt_kernel, adapt_const)
        # resmooth
        adapt_thresh_smooth = cv2.medianBlur(adapt_thresh, kernel_blur)

        mask_thresh = adapt_thresh_smooth

    closing = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    # try on bgRemoved?

    t2 = time.time()

    # set output
    if contour_smoothing < 0:
        final_mask = opening
    else:
        # contours
        # remove small items
        contours, cont_plot, hole_idxs = get_contours_complex(opening, 
                                            min_thresh=glob_thresh, 
                                           min_area=min_area, 
                                           contour_smoothing=contour_smoothing)

        # for some reason contours don't extend to the edge, so clip the edge
        # and resize
        mask_filt_raw = get_mask(mask_thresh, cont_plot, hole_idxs=hole_idxs)
        shape_tmp = mask_filt_raw.shape
        mask_filt1 = 200 * cv2.resize(mask_filt_raw[2:-2, 2:-2], shape_tmp).astype(np.uint8)
        # thresh and resmooth
        mask_filt = cv2.GaussianBlur(mask_filt1, (kernel_blur, kernel_blur), 0)
        #mask_filt = cv2.threshold(mask_filt2, glob_thresh, 1, cv2.THRESH_BINARY)
        final_mask = mask_filt

    t3 = time.time()
    # print ("Time to smooth contours:", t3-t2, "seconds")

    # skeletonize
    #medial = medial_axis(final_mask)
    #medial_int = medial.astype(np.uint8)
    medial_int = medial_axis(final_mask).astype(np.uint8)
    # print ("Time to compute medial_axis:", time.time() - t3, "seconds")
    # print ("Time to run dl_post_process_pred():", time.time() - t0, "seconds")

    return final_mask, medial_int


def cv2_skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object
    https://gist.github.com/jsheedy/3913ab49d344fac4d02bcc887ba4277d"""
    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def remove_sequential_duplicates(seq):
    # todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res


def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx+1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx-1]:v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_angle(p0, p1=np.array([0, 0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1) 
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)


def preprocess(img, thresh, img_mult=255, hole_size=300,
               cv2_kernel_close=7, cv2_kernel_open=7, verbose=True):
    '''
    http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_holes
    hole_size in remove_small_objects is the maximum area, in pixels of the
    hole
    '''

    # sometimes get a memory error with this approach
    if img.size < 10000000000:
        # if verbose:
        #     print("Run preprocess() with skimage")
        img = (img > (img_mult * thresh)).astype(np.bool)
        remove_small_objects(img, hole_size, in_place=True)
        remove_small_holes(img, hole_size, in_place=True)
        # img = cv2.dilate(img.astype(np.uint8), np.ones((7, 7)))

    # cv2 is generally far faster and more memory efficient (though less
    #  effective)
    else:
        # if verbose:
        #     print("Run preprocess() with cv2")

        #from road_raster.py, dl_post_process_pred() function
        kernel_close = np.ones((cv2_kernel_close, cv2_kernel_close), np.uint8)
        kernel_open = np.ones((cv2_kernel_open, cv2_kernel_open), np.uint8)
        kernel_blur = cv2_kernel_close

        # global thresh
        #mask_thresh = (img > (img_mult * thresh))#.astype(np.bool)
        blur = cv2.medianBlur( (img * img_mult).astype(np.uint8), kernel_blur)
        glob_thresh_arr = cv2.threshold(blur, thresh, 1, cv2.THRESH_BINARY)[1]
        glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)
        mask_thresh = glob_thresh_arr_smooth

        # opening and closing
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        #gradient = cv2.morphologyEx(mask_thresh, cv2.MORPH_GRADIENT, kernel)
        closing = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
        img = opening.astype(np.bool)
        #img = opening

    return img


def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


def visualize(img, G, vertices):
    plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (s, e) in G.edges():
        vals = flatten([[v] for v in G[s][e].values()])
        for val in vals:
            ps = val.get('pts', [])
            plt.plot(ps[:, 1], ps[:, 0], 'green')

    # draw node by o
    node, nodes = G.node(), G.nodes
    # deg = G.degree
    # ps = np.array([node[i]['o'] for i in nodes])
    ps = np.array(vertices)
    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    # title and show
    plt.title('Build Graph')
    plt.show()


def line_points_dist(line1, pts):
    return np.cross(
        line1[1] - line1[0],
        pts - line1[0]
    ) / np.linalg.norm(line1[1] - line1[0])


def remove_small_terminal(G):
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]
    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if s in terminal_points and val.get('weight', 0) < 10:
                G.remove_node(s)
            if e in terminal_points and val.get('weight', 0) < 10:
                G.remove_node(e)
    return


def add_small_segments(G,
                       terminal_points,
                       terminal_lines,
                       dist1=20,
                       dist2=100,
                       angle1=20,
                       angle2=160):
    node = G.node
    term = [node[t]['o'] for t in terminal_points]
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < dist1))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > dist1) & (dists < dist2))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > dist1:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if (-1*angle1 < angle < angle1) or (angle < -1*angle2) or (angle > angle2):
            good_pairs.append((s, e))

    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.node[s]['o'], G.node[e]['o']]
        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = G.node[s]['o'], G.node[e]['o']
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = '(' + ", ".join(line_strings) + ')'
            wkt.append(linestring.format(line))
    return wkt


def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps


def make_skeleton(img_loc,
                  thresh,
                  debug,
                  fix_borders,
                  replicate=5,
                  clip=2,
                  img_mult=255,
                  hole_size=300,
                  cv2_kernel_close=7,
                  cv2_kernel_open=7,
                  use_medial_axis=False,
                  num_classes=1,
                  skeleton_band='all'):
    '''
    Extract a skeleton from a mask.
    skeleton_band is the index of the band of the mask to use for
        skeleton extractionk, set to string 'all' to use all bands
    '''

    # print ("Executing make_skeleton...")
    t0 = time.time()
    #replicate = 5
    #clip = 2
    rec = replicate + clip

    # read in data
    if num_classes == 1:
        try:
            img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
        except:
            img = skimage.io.imread(img_loc, as_gray=True).astype(np.uint8)#[::-1]

    else:
        # ensure 8bit?
        img_tmp = skimage.io.imread(img_loc).astype(np.uint8)
        #img_tmp = skimage.io.imread(img_loc)
        # we want skimage to read in (channels, h, w) for multi-channel
        #   assume less than 20 channels
        if img_tmp.shape[0] > 20:
            img_full = np.moveaxis(img_tmp, 0, -1)
        else:
            img_full = img_tmp
        # select the desired band for skeleton extraction
        #  if < 0, sum all bands
        if type(skeleton_band) == str:  #skeleton_band < 0:
            img = np.sum(img_full, axis=0).astype(np.int8)
        else:
            img = img_full[skeleton_band, :, :]

    # potentially keep only subset of data
    shape0 = img.shape

    if fix_borders:
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate,
                                 replicate, cv2.BORDER_REPLICATE)
    img_copy = None
    if debug:
        if fix_borders:
            img_copy = np.copy(img[replicate:-replicate,replicate:-replicate])
        else:
            img_copy = np.copy(img)

    t1 = time.time()
    img = preprocess(img, thresh, img_mult=img_mult, hole_size=hole_size,
                     cv2_kernel_close=cv2_kernel_close,
                     cv2_kernel_open=cv2_kernel_open)
    t2 = time.time()
    if not np.any(img):
        return None, None

    if not use_medial_axis:
        ske = skeletonize(img).astype(np.uint16)
        t3 = time.time()

    else:
        ske = skimage.morphology.medial_axis(img).astype(np.uint16)
        t3 = time.time()

    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
        t4 = time.time()

    t1 = time.time()
    return img, ske


def build_graph_wkt(img_loc, out_ske_file, out_gpickle='', thresh=0.3,
                debug=False, add_small=True, fix_borders=True,
                skel_replicate=5, skel_clip=2, min_subgraph_length_pix=150,
                img_mult=255, hole_size=300, cv2_kernel_close=7, cv2_kernel_open=7,
                num_classes=1,
                skeleton_band='all',
                verbose=False):

    # create skeleton
    img_copy, ske = make_skeleton(img_loc,
                                  thresh,
                                  debug,
                                  fix_borders,
                                  replicate=skel_replicate,
                                  clip=skel_clip,
                                  img_mult=img_mult,
                                  hole_size=hole_size,
                                  cv2_kernel_close=cv2_kernel_close,
                                  cv2_kernel_open=cv2_kernel_open,
                                  skeleton_band=skeleton_band,
                                  num_classes=num_classes)
    if ske is None:
        return [linestring.format("EMPTY")]
    # save to file
    if out_ske_file:
        cv2.imwrite(out_ske_file, ske.astype(np.uint8)*255)

    # create graph
    if np.max(ske.shape) > 32767:
        assert False
    else:
        G = sknw.build_sknw(ske, multi=True)
    remove_small_terminal(G)
    if len(G.edges()) == 0:
        return [linestring.format("EMPTY")]

    if verbose:
        node_tmp= list(G.nodes())[-1]
        edge_tmp = list(G.edges())[-1]

    t01 = time.time()
    G = clean_sub_graphs(G, min_length=min_subgraph_length_pix,
                     max_nodes_to_skip=100,
                     weight='weight', verbose=verbose,
                     super_verbose=False)
    t02 = time.time()
    # save G
    if len(out_gpickle) > 0:
        nx.write_gpickle(G, out_gpickle)

    node_lines = graph2lines(G)
    if not node_lines:
        return [linestring.format("EMPTY")]

    node = G.node
    deg = dict(G.degree())
    wkt = []
    terminal_points = [i for i, d in deg.items() if d == 1]

    # refine wkt
    # print ("Refine wkt...")
    terminal_lines = {}
    vertices = []
    for i,w in enumerate(node_lines):
        if ((i % 10000) == 0) and (i > 0):
            print ("  ", i, "/", len(node_lines))
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):

                s_coord, e_coord = node[s]['o'], node[e]['o']
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)
        for coord_list in segments:
            if len(coord_list) > 1:
                line = '(' + ", ".join(coord_list) + ')'
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format('(' + line + ')'))

    if add_small and len(terminal_points) > 1:
        wkt.extend(add_small_segments(G, terminal_points, terminal_lines))

    if debug:
        vertices = flatten(vertices)
        visualize(img_copy, G, vertices)

    if not wkt:
        return [linestring.format("EMPTY")]

    return wkt


def _build_graph_wkt_iterable(args):
    (
        imfile,
        im_prefix,
        indir,
        spacenet_naming_convention,
        out_ske_dir,
        out_gdir,
        thresh,
        debug,
        add_small,
        fix_borders,
        skel_replicate,
        skel_clip,
        img_mult,
        hole_size,
        cv2_kernel_close,
        cv2_kernel_open,
        min_subgraph_length_pix,
        num_classes,
        skeleton_band,
    ) = args

    t1 = time.time()
    img_loc = os.path.join(indir, imfile)

    if spacenet_naming_convention:
        im_root = 'AOI' + imfile.split('AOI')[-1].split('.')[0]
    else:
        im_root = imfile.split('.')[0]
    if len(im_prefix) > 0:
        im_root = im_root.split(im_prefix)[-1]

    if out_ske_dir:
        out_ske_file = os.path.join(out_ske_dir, imfile)
    else:
        out_ske_file = ''

    if len(out_gdir) > 0:
        out_gpickle = os.path.join(out_gdir, imfile.split('.')[0] + '.gpickle')
    else:
        out_gpickle = ''

    # create wkt list
    wkt_list = build_graph_wkt(img_loc, out_ske_file,
            out_gpickle=out_gpickle, thresh=thresh,
            debug=debug, add_small=add_small, fix_borders=fix_borders,
            skel_replicate=skel_replicate, skel_clip=skel_clip,
            img_mult=img_mult, hole_size=hole_size,
            cv2_kernel_close=cv2_kernel_close, cv2_kernel_open=cv2_kernel_open,
            min_subgraph_length_pix=min_subgraph_length_pix,
            num_classes=num_classes,
            skeleton_band=skeleton_band)
    return (im_root, wkt_list)


def build_wkt_dir(indir, outfile, out_ske_dir, out_gdir='', thresh=0.3,
                  im_prefix='',
                  debug=False, add_small=True, fix_borders=True,
                  skel_replicate=5, skel_clip=2,
                  img_mult=255,
                  hole_size=300, cv2_kernel_close=7, cv2_kernel_open=7,
                  min_subgraph_length_pix=50,
                  spacenet_naming_convention=False,
                  num_classes=1,
                  skeleton_band='all'):
    '''Execute built_graph_wkt for an entire folder
    Split image name on AOI, keep only name after AOI.  This is necessary for
    scoring'''

    all_data = []
    im_files = np.sort([z for z in os.listdir(indir) if z.endswith('.tif')])
    nfiles = len(im_files)

    print(indir, nfiles)

    args_list = []
    for i, imfile in tqdm.tqdm(enumerate(im_files), total=nfiles):
        args = (
            imfile,
            im_prefix,
            indir,
            spacenet_naming_convention,
            out_ske_dir,
            out_gdir,
            thresh,
            debug,
            add_small,
            fix_borders,
            skel_replicate,
            skel_clip,
            img_mult,
            hole_size,
            cv2_kernel_close,
            cv2_kernel_open,
            min_subgraph_length_pix,
            num_classes,
            skeleton_band,
        )
        args_list.append(args)

    with Pool(cpu_count()) as p:
        data = list(tqdm.tqdm(
            iterable=p.imap_unordered(_build_graph_wkt_iterable, args_list),
            total=len(args_list)))

    for im_root, wkt_list in sorted(data):
        for v in wkt_list:
            all_data.append((im_root, v))

    # save to csv
    df = pd.DataFrame(all_data, columns=['ImageId', 'WKT_Pix'])
    df.to_csv(outfile, index=False)

    return df


def run_skeletonize(conf):
    spacenet_naming_convention = False # True

    preds_dirname = conf.modelname.replace('_th06', '')
    print('preds', preds_dirname)

    im_dir = "{}{}/{}/".format(
        "/wdata", "/working/sp5r2/models/preds", preds_dirname)
    im_prefix = ''

    if conf.num_folds > 1:
        im_dir = im_dir + "merged_test"
    else:
        im_dir = im_dir + "fold0_test"
        im_prefix = 'fold0_'

    os.makedirs(im_dir, exist_ok=True)

    # outut csv file
    outfile_csv = "{}/working/sp5r2/models/wkt/{}/wkt_submission_nospeed.csv".format(
        "/wdata", conf.modelname)
    Path(outfile_csv).parent.mkdir(parents=True, exist_ok=True)

    # output ske
    out_ske_dir = "{}/working/sp5r2/models/ske/{}".format(
        "/wdata", conf.modelname)
    Path(out_ske_dir).mkdir(parents=True, exist_ok=True)

    # output pkl
    out_gdir = "{}/working/sp5r2/models/sknw_gpickle/{}".format(
        "/wdata", conf.modelname)
    Path(out_gdir).mkdir(parents=True, exist_ok=True)

    thresh = conf.skeleton_thresh
    min_subgraph_length_pix = conf.min_subgraph_length_pix

    debug=False
    add_small=True
    fix_borders=True
    skel_replicate=5
    skel_clip=2
    img_mult=255
    hole_size=300
    cv2_kernel_close=7
    cv2_kernel_open=7

    logger.info("Building wkts...")
    t0 = time.time()
    df = build_wkt_dir(im_dir, outfile_csv, out_ske_dir, out_gdir, thresh,
                       debug=debug,
                       add_small=add_small,
                       fix_borders=fix_borders,
                       skel_replicate=skel_replicate,
                       skel_clip=skel_clip,
                       img_mult=img_mult,
                       hole_size=hole_size,
                       min_subgraph_length_pix=min_subgraph_length_pix,
                       cv2_kernel_close=cv2_kernel_close,
                       cv2_kernel_open=cv2_kernel_open,
                       skeleton_band=conf.skeleton_band,
                       num_classes=conf.num_classes,
                       im_prefix=im_prefix,
                       spacenet_naming_convention=spacenet_naming_convention)

    t1 = time.time()
    logger.info("len df: {}".format(len(df)))
    logger.info("outfile: {}".format(outfile_csv))
    logger.info("Total time to run build_wkt_dir: {} seconds".format(t1-t0))
