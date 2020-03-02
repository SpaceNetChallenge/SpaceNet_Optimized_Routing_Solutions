import argparse

import numpy
import cv2
import torch
import power_fist

from src.dataset_v2 import SpacenetDataInterface

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

numpy.seterr(divide='ignore', invalid='ignore')  # FIXME ignore non-stop warnings from albumentations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_default.yml',
        help='Path to training config file',
    )
    parser.add_argument(
        '--paths',
        type=str,
        default='configs/paths_default.yml',
        help='Path to config file with paths for dataset and logs',
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=-1,
        help='Fold number to train and evaluate on',
    )
    parser.add_argument(
        '--test-loader',
        dest='test_loader',
        action='store_true',
        help='Dump images from dataloader before starting training',
    )
    parser.add_argument(
        '--no-train',
        dest='train',
        action='store_false',
        help='Skip training stage of the pipeline, evaluate the model',
    )
    parser.add_argument(
        '--no-predict',
        dest='predict',
        action='store_false',
        help='Skip evaluation of the model',
    )
    parser.add_argument(
        '--resume',
        dest='resume',
        action='store_true',
        help='Resume training by loading the best checkpoints from dumps',
    )
    parser.add_argument(
        '--benchmark',
        dest='benchmark',
        action='store_true',
        help='Run speed benchmark using the model from the config',
    )
    parser.set_defaults(
        test_loader=False,
        train=True,
        predict=True,
        resume=False,
        benchmark=False,
    )
    args = parser.parse_args()
    return args


def main():
    power_fist.utils.set_global_seeds(42)
    args = parse_args()

    config = power_fist.config_utils.get_config(args.config)
    paths_config = power_fist.config_utils.get_paths(args.paths)
    if args.fold >= 0:
        config['data_params']['fold'] = args.fold

    stage_templates = power_fist.config_utils.get_stage_templates()
    for stage in config['stages']:
        if isinstance(config['stages'][stage], str):
            config['stages'][stage] = stage_templates[config['stages'][stage]]

    runner = power_fist.pipeline.Runner(
        config=config,
        paths_config=paths_config,
        data_interface_class=SpacenetDataInterface,
    )

    if args.test_loader:
        runner.test_loader()

    if args.benchmark:
        runner.run_benchmark()

    if args.train:
        if not args.resume:
            runner.run_training()
        else:
            runner.resume_training()
    if args.predict:
        runner.predict()


if __name__ == '__main__':
    main()
