import argparse
from typing import List, Tuple, Dict
import glob
from tqdm import tqdm
import warnings
import multiprocessing

import cv2
import numpy as np
import pandas as pd
import networkx as nx
from simplification.cutil import simplify_coords

from pygeoif import LineString
from scipy import ndimage
from scipy.ndimage import binary_dilation
import sknw
from shapely import wkt
from shapely.geometry import LineString
from skimage.filters import gaussian
from skimage.morphology import remove_small_objects, skeletonize

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.lines as mlines

warnings.filterwarnings('ignore')  # non-stop warning from numba

MEAN_SPEED = 11.15411529750582  # mps


def newline(ax: Axes, p1: List[float], p2: List[float]):
    l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]])
    ax.add_line(l)
    return l


def prepare_mask(mask: np.ndarray, sigma: float = 0.5, threshold: float = 0.3, small_obj_size: int = 300,
                    dilation: int = 1) -> np.ndarray:
    assert mask.ndim == 3
    assert mask.shape[0] == 1300 and mask.shape[1] == 1300, 'Some strange things in code for not 1300x1300 shapes'
    mask = gaussian(mask, sigma=sigma)
    mask = mask[..., 0]
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 1
    mask = np.array(mask, dtype="uint8")
    mask = mask[:1300, :1300]  # ????
    mask = cv2.copyMakeBorder(mask, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    if dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)
    mask, _ = ndimage.label(mask)
    mask = remove_small_objects(mask, small_obj_size)
    mask[mask > 0] = 1
    return mask


def prepare_graph_edges(graph: nx.Graph) -> List:
    node, nodes = graph.node, graph.nodes()
    all_coords = []
    # draw edges by pts
    for (s, e) in graph.edges():
        for k in range(len(graph[s][e])):
            ps = graph[s][e][k]['pts']
            coords = []
            start = (int(nodes[s]['o'][1]), int(nodes[s]['o'][0]))
            all_points = set()

            for i in range(1, len(ps)):
                pt1 = (int(ps[i - 1][1]), int(ps[i - 1][0]))
                pt2 = (int(ps[i][1]), int(ps[i][0]))
                if pt1 not in all_points and pt2 not in all_points:
                    coords.append(pt1)
                    all_points.add(pt1)
                    coords.append(pt2)
                    all_points.add(pt2)
            end = (int(nodes[e]['o'][1]), int(nodes[e]['o'][0]))

            same_order = True
            if len(coords) > 1:
                same_order = np.math.hypot(start[0] - coords[0][0],
                                           start[1] - coords[0][1]) <= np.math.hypot(end[0] - coords[0][0],
                                                                                     end[1] - coords[0][1])
            if same_order:
                coords.insert(0, start)
                coords.append(end)
            else:
                coords.insert(0, end)
                coords.append(start)
            coords = simplify_coords(coords, 2.0)
            all_coords.append(coords)
    return all_coords


def get_line_strings_from_coords(all_coords: List) -> List:
    lines = []
    for coords in all_coords:
        if len(coords) > 0:
            line_obj = LineString(coords)
            lines.append(line_obj)

    new_lines = remove_duplicates(lines)
    new_lines = filter_lines(new_lines, calculate_node_count(new_lines))
    return [l.wkt for l in new_lines]


def mask_to_line_strings(mask: np.ndarray, sigma: float = 0.5, threshold: float = 0.3, small_obj_size: int = 300,
                         dilation: int = 1):
    mask = prepare_mask(mask=mask, sigma=sigma, threshold=threshold, small_obj_size=small_obj_size, dilation=dilation)

    ske = np.array(skeletonize(mask), dtype="uint8")
    ske = ske[8:-8, 8:-8]
    graph = sknw.build_sknw(ske, multi=True)
    all_coords = prepare_graph_edges(graph)
    line_strings = get_line_strings_from_coords(all_coords)
    return line_strings


def remove_duplicates(lines: List) -> List:
    all_paths = set()
    new_lines = []
    for l, line in enumerate(lines):
        points = line.coords
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            if (pt1, pt2) not in all_paths and (pt2, pt1) not in all_paths and not pt1 == pt2:
                new_lines.append(LineString((pt1, pt2)))
                all_paths.add((pt1, pt2))
                all_paths.add((pt2, pt1))
    return new_lines


def filter_lines(new_lines: List, node_count: Dict) -> List:
    filtered_lines = []
    for line in new_lines:
        points = line.coords
        pt1 = (int(points[0][0]), int(points[0][1]))
        pt2 = (int(points[1][0]), int(points[1][1]))

        length = np.math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

        if not ((node_count[pt1] == 1 and node_count[pt2] > 2  or node_count[pt2] == 1 and node_count[pt1] > 2) and length < 10):
            filtered_lines.append(line)
    return filtered_lines


def calculate_node_count(new_lines: List):
    node_count = {}
    for l, line in enumerate(new_lines):
        points = line.coords
        for i in range(1, len(points)):
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            pt1c = node_count.get(pt1, 0)
            pt1c += 1
            node_count[pt1] = pt1c
            pt2c = node_count.get(pt2, 0)
            pt2c += 1
            node_count[pt2] = pt2c
    return node_count


def get_dist_and_time_from_line(line_str: str, mask_h: int, mask_w: int) -> Tuple[float, float]:
    line_str = line_str[12:-1]  # remove brackets
    p1, p2 = line_str.split(',')
    p1 = np.array([float(coord) for coord in p1.split(' ')])
    p2 = np.array([float(coord) for coord in p2.split(' ')[1:]])  # remove empty string at index 0
    p1[0] /= mask_w
    p2[0] /= mask_w
    p1[1] /= mask_h
    p2[1] /= mask_h

    p1 *= 400  # 400x400 m
    p2 *= 400

    diff = p1 - p2
    dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
    time = dist / MEAN_SPEED

    return dist, time


def get_mask_id(mask_path: str) -> str:
    path_splitted = np.array(mask_path.split('_'))
    mask_num = path_splitted[-1].split('.')[0]
    split_start = np.argwhere(path_splitted == 'AOI')[0][0]
    split_end = np.argwhere(path_splitted == 'PS-RGB')[0][0]
    id_ = '_'.join(path_splitted[split_start:split_end].tolist() + [mask_num])
    return id_


def get_predictions_from_mask(mask_path: str) -> List[List]:
    mask = cv2.imread(mask_path)
    mask_id = get_mask_id(mask_path)
    if mask.shape[0] != 1300 or mask.shape[1] != 1300:
        raise NotImplemented('Implemented only for masks with sizes 1300x1300')

    image_predictions = []

    line_strings = mask_to_line_strings(mask=mask)
    if len(line_strings) == 0:
        return [[mask_id, wkt.loads('LINESTRING EMPTY').wkt, 0, 0]]
    for line_str in line_strings:
        dist, time = get_dist_and_time_from_line(line_str=line_str, mask_h=mask.shape[0], mask_w=mask.shape[1])
        image_predictions.append([mask_id, line_str, dist, time])

    return image_predictions


def get_predictions_from_folder(masks_folder: str) -> pd.DataFrame:
    masks_paths = glob.glob(f"{masks_folder}/*.png")

    with multiprocessing.Pool(16) as pool:
        predictions = list(tqdm(
            pool.imap_unordered(get_predictions_from_mask, masks_paths),
            total=len(masks_paths),
            desc='Converting dataset...',
        ))

    predictions = [pred for mask_pred in predictions for pred in mask_pred]

    predictions_df = pd.DataFrame(predictions, columns=['ImageId', 'WKT_Pix', 'length_m', 'travel_time_s'])
    return predictions_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--masks-folder',
        type=str,
        required=True,
        help='Path to a directory with predicted masks of size 1300x1300 and png format',
    )
    parser.add_argument(
        '--save-path',
        type=str,
        required=True,
        help='Path to save (with name of csv file)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_df = get_predictions_from_folder(args.masks_folder)
    print(predictions_df.head())
    predictions_df.to_csv(args.save_path, index=False)
    print(f'Submission saved to {args.save_path}')


if __name__ == "__main__":
    main()
