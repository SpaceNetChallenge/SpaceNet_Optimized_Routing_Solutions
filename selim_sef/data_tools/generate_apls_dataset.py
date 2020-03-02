import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import random
from functools import partial
from multiprocessing.pool import Pool
import cv2
import numpy as np
import tqdm
from albumentations.augmentations.functional import random_crop


def generate_oof_masks(mask_file, img_dir="masks", oof_predictions_dir=None, img_size=640, masks_per_file=10):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(img_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "solution"), exist_ok=True)
    mask_dir = os.path.join("/home/selim/datasets/spacenet/train_mask_binned")
    pred_dir = os.path.join(oof_predictions_dir, "full_oof")
    mask = cv2.imread(os.path.join(mask_dir, mask_file[:-4] .replace("RGB", "MS")+ ".tif"), cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 255
    solution_mask = cv2.imread(os.path.join(pred_dir, mask_file[:-4]+".png"), cv2.IMREAD_GRAYSCALE)
    id = mask_file[:-4]
    for i in range(masks_per_file):
        h_start, w_start = random.random(), random.random()
        crop = random_crop(mask, img_size, img_size, h_start, w_start)
        if np.sum(crop) < 2000 * 255 and random.random() < 0.9:
            continue

        solution_crop = random_crop(solution_mask, img_size, img_size, h_start, w_start)
        cv2.imwrite(os.path.join(img_dir, "original", "{}_{}oof.png".format(id, i)), crop)
        cv2.imwrite(os.path.join(img_dir, "solution", "{}_{}oof.png".format(id, i)), solution_crop)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Synthetic Mask Generator")
    arg = parser.add_argument
    arg('--out_dir', default='/home/selim/datasets/spacenet/masks_apls')
    arg('--workers', type=int, default=12)
    arg('--oof_predictions_dir', type=str, default="/home/selim/kaggle/oof_preds/")
    args = parser.parse_args()

    mask_files = os.listdir(os.path.join(args.oof_predictions_dir, "full_oof"))
    with Pool(processes=args.workers) as p:
        with tqdm.tqdm(total=len(mask_files)) as pbar:
            for i, v in tqdm.tqdm(enumerate(p.imap_unordered(partial(generate_oof_masks, img_dir=args.out_dir, oof_predictions_dir=args.oof_predictions_dir, masks_per_file=10), mask_files))):
                pbar.update()
