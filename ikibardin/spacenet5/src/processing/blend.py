import os
import glob
import argparse
import multiprocessing
from typing import Dict, List, Union

import numpy as np
import cv2
from tqdm import tqdm

BLEND_CONFIG = {
    'rx101_FIXED_1': {
        'weight': 0.6,
        'folds': [0, 1],
    },
    'rn50_fpn_FIXED_0': {
        'weight': 0.2,
        'folds': [2, 3],
    },
    'srx50_FIXED_0': {
        'weight': 0.2,
        'folds': [4],
    },
}

BlendConfigType = Dict[str, Dict[str, Union[float, List[int]]]]


def validate_config(config: BlendConfigType):
    total_weight = 0.0
    for _, params in config.items():
        total_weight += params['weight']
    assert np.allclose(total_weight, 1.0), total_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dumps',
        type=str,
        required=True,
        help='Path to learning dumps directory',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Where to save blended masks',
    )
    return parser.parse_args()


class Blender:
    def __init__(self, dumps_dir: str, output_dir: str, blend_config: BlendConfigType):
        self._dumps_dir = dumps_dir
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._config = blend_config

    def __call__(self, image_id: str):
        blended_mask = self._get_blended_mask(image_id)
        cv2.imwrite(self._get_output_path(image_id), blended_mask)

    def _get_output_path(self, image_id: str) -> str:
        return os.path.join(self._output_dir, image_id)

    def _get_blended_mask(self, image_id: str) -> np.ndarray:
        masks = []
        weights = []
        for model_name, params in self._config.items():
            masks.append(self._get_average_mask_for_model(model_name, image_id))
            weights.append(params['weight'])
        blended_mask = np.average(masks, axis=0, weights=weights)
        blended_mask[blended_mask > 1.0] = 1.0
        return (blended_mask * 255.0).astype(np.uint8)

    def _get_average_mask_for_model(self, model_name: str, image_id: str) -> np.ndarray:
        masks = []
        for fold_id in self._config[model_name]['folds']:
            masks.append(self._load_mask(model_name, fold_id, image_id))
        return np.mean(masks, axis=0)

    def _load_mask(self, model_name: str, fold_id: int, image_id: str) -> np.ndarray:
        path_pattern = os.path.join(self._dumps_dir, model_name, f'fold_{fold_id}', 'predictions',
                                    '*', 'test', image_id)
        candidates = glob.glob(path_pattern)
        assert len(candidates) == 3, candidates
        masks = []
        for path in candidates:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            assert mask is not None, path
            assert len(mask.shape) == 2, mask.shape
            mask = mask.astype(np.float64) / 255.0
            masks.append(mask)
        return np.mean(masks, axis=0)


def main():
    validate_config(BLEND_CONFIG)
    args = parse_args()
    sample_model = list(BLEND_CONFIG.keys())[0]
    path_pattern = os.path.join(args.dumps, sample_model,
                                f'fold_{BLEND_CONFIG[sample_model]["folds"][0]}',
                                'predictions',
                                '*', 'test', '*')
    image_ids = set([os.path.basename(path) for path in glob.glob(path_pattern)])

    if len(image_ids) == 0:
        raise FileNotFoundError(path_pattern)

    print(f'Found {len(image_ids)} unique image ids')

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        list(tqdm(
            pool.imap_unordered(Blender(args.dumps, args.output, BLEND_CONFIG), image_ids),
            total=len(image_ids),
            desc=f'Saving blended masks to `{args.output}`',
        ))


if __name__ == '__main__':
    main()
