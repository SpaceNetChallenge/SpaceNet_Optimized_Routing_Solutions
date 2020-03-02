import argparse
import glob
import os

import skimage.io
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from scipy.spatial.distance import euclidean

from functools import partial
from multiprocessing.pool import Pool

import cv2
import pandas as pd

import numpy as np
import pygeoif
import tqdm


def stretch_8bit(bands, lower_percent=1, higher_percent=99):
    out = np.zeros_like(bands).astype(np.float32)
    for i in range(bands.shape[-1]):
        a = 0
        b = 1
        band = bands[:, :, i].flatten()
        filtered = band[band > 0]
        if len(filtered) == 0:
            continue
        c = np.percentile(filtered, lower_percent)
        d = np.percentile(filtered, higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)


def convert_image(tif_image, dataset_version):
    if dataset_version > 3:
        return tif_image
    else:
        return stretch_8bit(tif_image) * 255


def create_mask(lines, thickness=16):
    mask = np.zeros((1300, 1300))
    for line in lines:
        wkt_pix = line["wkt_pix"]
        if "EMPTY" not in wkt_pix:
            line = wkt_pix
            points = pygeoif.from_wkt(line).coords
            for i in range(1, len(points)):
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(mask, pt1, pt2, (1,), thickness=thickness)
    return mask * 255


def create_speed_mask(lines, thickness=16):
    max_speed = 35  # mps

    mask = np.zeros((1300, 1300))
    for line in lines:
        wkt_pix = line["wkt_pix"]
        length_m = line["length_m"]
        travel_time_s = line["travel_time_s"]
        if "EMPTY" not in wkt_pix:
            speed = 9. if travel_time_s == 0 else length_m / travel_time_s
            speed_normalized = int(255 * speed / max_speed)

            line = wkt_pix
            wkt = pygeoif.from_wkt(line)
            points = wkt.coords
            for i in range(1, len(points)):
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(mask, pt1, pt2, (speed_normalized,), thickness=thickness)
    return mask


def write_mask(id_to_lines, out_dir):
    id, lines = id_to_lines
    mask = create_mask(lines)
    cv2.imwrite(os.path.join(out_dir, "{}.png".format(id)), mask)


def write_speed_mask(id_to_lines, out_dir):
    id, lines = id_to_lines
    mask = create_speed_mask(lines)
    cv2.imwrite(os.path.join(out_dir, "{}.png".format(id)), mask)


def write_image(img_path, out_dir, dataset_version):
    img = skimage.io.imread(img_path)
    img = convert_image(img, dataset_version)
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(out_dir, "{}.png".format(img_id)), img[..., ::-1])

def write_junctions(id_to_lines, out_dir):
    id, lines = id_to_lines
    mask = create_mask_junctions(lines)
    cv2.imwrite(os.path.join(out_dir, "{}.png".format(id)), mask)


def ds_point(p):
    return (p[0] // 5, p[1] // 5)


def create_mask_junctions(lines, thickness=16):
    mask = np.zeros((1300, 1300))
    point_map = {}
    for line in lines:
        wkt_pix = line["wkt_pix"]
        if "EMPTY" not in wkt_pix:
            line = wkt_pix
            points = pygeoif.from_wkt(line).coords
            for i in range(1, len(points)):
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                point_map[ds_point(pt1)] = point_map.get(ds_point(pt1), 0) + 1
                point_map[ds_point(pt2)] = point_map.get(ds_point(pt2), 0) + 1

    for line in lines:
        wkt_pix = line["wkt_pix"]
        if "EMPTY" not in wkt_pix:
            line = wkt_pix
            points = pygeoif.from_wkt(line).coords
            for i in range(1, len(points)):
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                pt2_ori = pt2
                if point_map[ds_point(pt1)] < 3 and point_map[ds_point(pt2)] < 3:
                    continue
                if point_map[ds_point(pt1)] >= 3:
                    distance = euclidean(pt1, pt2)
                    if distance > 32:
                        frac = 32 / distance
                        xc = pt1[0] + (pt2[0] - pt1[0]) * frac
                        yc = pt1[1] + (pt2[1] - pt1[1]) * frac
                        pt2 = (int(xc), int(yc))
                    cv2.line(mask, pt1, pt2, (1,), thickness=thickness)
                pt2 = pt2_ori
                if point_map[ds_point(pt2)] >= 3:
                    tmp = pt1
                    pt1 = pt2
                    pt2 = tmp
                    distance = euclidean(pt1, pt2)
                    if distance > 32:
                        frac = 32 / distance
                        xc = pt1[0] + (pt2[0] - pt1[0]) * frac
                        yc = pt1[1] + (pt2[1] - pt1[1]) * frac
                        pt2 = (int(xc), int(yc))
                    cv2.line(mask, pt1, pt2, (1,), thickness=thickness)

    return mask * 255

def process_dataset(dataset_path, output_dir, workers=12):
    junction_mask_dir = os.path.join(output_dir, "junction_masks")
    os.makedirs(junction_mask_dir, exist_ok=True)
    csvs = glob.glob(os.path.join(dataset_path, "*simp.csv"))
    id_to_data = {}
    for csv in csvs:
        rows = pd.read_csv(csv).values
        for row in rows:
            imageid, wkt_pix, length_m, travel_time_s = row
            lines = id_to_data.get(imageid, [])
            lines.append({
                "wkt_pix": wkt_pix,
                "length_m": length_m,
                "travel_time_s": travel_time_s,
            })
            id_to_data[imageid] = lines
    print(len(id_to_data))
    with Pool(processes=workers) as p:
        with tqdm.tqdm(total=len(id_to_data)) as pbar:
            for _ in tqdm.tqdm(enumerate(p.imap_unordered(partial(write_junctions, out_dir=junction_mask_dir), id_to_data.items()))):
                pbar.update()



if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser("Prepare masks")
        arg = parser.add_argument
        arg('--data-dirs', nargs='+')
        arg('--out-dir', type=str, default="/wdata")
        args = parser.parse_args()
        for data_dir in args.data_dirs:
            process_dataset(data_dir, output_dir=args.out_dir)