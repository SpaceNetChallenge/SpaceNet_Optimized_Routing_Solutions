import os
import multiprocessing
from typing import Tuple, Dict

import pandas as pd

import torch
from torch.utils.data import sampler, DataLoader, Dataset
from tqdm import tqdm

from power_fist.common_utils.segmentation_dataset import SegmentationBatchHandler
from power_fist.common_utils.saver import PredictionSaverInterface


class DataInterface:
    """An abstract class for data interface
    """

    def __init__(self, paths_config: Dict, data_params: Dict, predict_params: Dict):
        self._paths_config = paths_config
        self._data_params = data_params
        self._predict_params = predict_params

    def make_train_valid_loaders(self) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    def make_test_loader(self, part: str, saver_: PredictionSaverInterface) -> DataLoader:
        raise NotImplementedError

    def make_batch_handler(self, training: bool) -> SegmentationBatchHandler:
        raise NotImplementedError

    def make_predictions_saver(
        self, experiment_name: str, checkpoint_path: str, dataset_part: str
    ) -> PredictionSaverInterface:
        raise NotImplementedError


class SegmentationDataInterface(DataInterface):
    """An abstract class
    """

    def __init__(
        self,
        paths_config: Dict,
        data_params: Dict,
        predict_params: Dict,
        segmentation_dataset_class,
        fold_column_name: str = 'fold_id',
    ):
        super().__init__(paths_config, data_params, predict_params)
        self._segmentation_dataset_class = segmentation_dataset_class
        self._fold_column_name = fold_column_name

    def make_train_valid_loaders(self, distributed=False) -> Tuple[DataLoader, DataLoader]:
        train_dataset, valid_dataset = self.make_train_valid_datasets()

        train_weights = torch.DoubleTensor([1.0] * len(train_dataset))  # uniform sampling
        train_sampler = sampler.WeightedRandomSampler(
            weights=train_weights,
            num_samples=self._data_params['batch_size'] * self._data_params['steps_per_epoch'],
        )
        train_loader = self._make_loader(train_dataset, train_sampler, mode='train', distributed=distributed)
        valid_loader = self._make_loader(
            valid_dataset,
            sampler.SequentialSampler(valid_dataset),
            mode='valid',
            distributed=distributed,
        )
        return train_loader, valid_loader

    def make_test_loader(self, part, saver_) -> DataLoader:
        test_dataset = self._make_test_dataset(part, saver_)
        test_loader = self._make_loader(test_dataset, sampler.SequentialSampler(test_dataset), mode='test')
        return test_loader

    def make_batch_handler(self, training: bool) -> SegmentationBatchHandler:
        if training or 'TTA' not in self._predict_params.keys():
            return SegmentationBatchHandler(training, num_tta=None)
        elif self._predict_params['TTA'] == 'D4':
            return SegmentationBatchHandler(training, num_tta=8)
        elif self._predict_params['TTA'] == 'HorFlip':
            return SegmentationBatchHandler(training, num_tta=2)
        elif self._predict_params['TTA'] == 'TTA2':
            return SegmentationBatchHandler(training, num_tta=2)
        elif self._predict_params['TTA'] == 'TTA10':
            return SegmentationBatchHandler(training, num_tta=10)
        else:
            raise ValueError('Unknown TTA "{}"'.format(self._predict_params['TTA']))

    def make_predictions_saver(self, experiment_name, checkpoint_path, dataset_part):
        raise NotImplementedError

    def _get_predictions_destination(self, experiment_name: str, checkpoint_path: str, dataset_part: str) -> str:
        stage_name = checkpoint_path.split('/')[-3]
        checkpoint_name = 'fold{}_{}_{}'.format(
            self._data_params['fold'],
            stage_name,
            checkpoint_path.split('/')[-1].split('.pth')[-2],
        )
        return os.path.join(
            self._paths_config['dumps']['path'],
            experiment_name,
            'fold_{}'.format(self._data_params['fold']),
            self._paths_config['dumps']['predictions'],
            checkpoint_name,
            dataset_part,
        )

    def make_train_valid_datasets(self) -> Tuple[Dataset, Dataset]:
        train_df, valid_df = self._make_train_valid_dataframes()
        train_dataset = self._segmentation_dataset_class(
            self._paths_config,
            self._data_params,
            train_df,
            self._make_train_transform(),
            mode='train',
        )
        valid_dataset = self._segmentation_dataset_class(
            self._paths_config,
            self._data_params,
            valid_df,
            self._make_valid_transform(),
            mode='valid',
        )
        return train_dataset, valid_dataset

    def _make_test_dataset(self, part: str, saver_: PredictionSaverInterface) -> Dataset:
        test_df = self._make_test_dataframe(part, saver_)
        test_dataset = self._segmentation_dataset_class(
            self._paths_config,
            self._data_params,
            test_df,
            self._make_test_transform(),
            mode='test',
        )
        return test_dataset

    def _make_loader(self, dataset: Dataset, sampler_: sampler.Sampler, mode: str, distributed=False) -> DataLoader:
        assert mode in ('train', 'valid', 'test')
        bs = (self._data_params['batch_size'] if mode == 'train' else self._predict_params['batch_size'])

        # if mode == 'test' and 'TTA' in self._predict_params:
        #     bs //= 8
        # if isinstance(sampler_, sampler.SequentialSampler):
        #    bs = 50

        if distributed:
            sampler_ = torch.utils.data.distributed.DistributedSampler(dataset)

        return DataLoader(
            dataset=dataset,
            batch_size=bs,
            sampler=sampler_,
            num_workers=self._data_params['num_workers'],
        )

    def _make_train_valid_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self._data_params['folds_csv'])

        valid_items_mask = df[self._fold_column_name] == self._data_params['fold']

        valid_df = df[valid_items_mask].reset_index(drop=True)
        train_df = df[~valid_items_mask].reset_index(drop=True)

        if 'pseudo_csv' in self._data_params:
            pseudo_df = pd.read_csv(self._data_params['pseudo_csv'])
            train_df = train_df.append(pseudo_df).reset_index(drop=True)

        print(f'train samples:{train_df.shape[0]}; valid samples:{valid_df.shape[0]}')
        return train_df, valid_df

    def _make_test_dataframe(self, part: str, saver_: PredictionSaverInterface) -> pd.DataFrame:
        if part == 'test':
            df = pd.read_csv(self._predict_params['test_csv'])
        else:
            df_filename = self._data_params['folds_csv']
            if 'predict_csv' in self._predict_params:
                df_filename = self._predict_params['predict_csv']
            train_df = pd.read_csv(df_filename)
            valid_items_mask = (train_df[self._fold_column_name] == self._data_params['fold'])

            if part == 'val':
                df = train_df[valid_items_mask].reset_index(drop=True)
            elif part == 'train':
                df = train_df[~valid_items_mask].reset_index(drop=True)
            else:
                raise ValueError('Unknown part "{}". Expected to be one of "test", "val", "train"'.format(part))

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            already_exist_mask = pd.Series(
                list(
                    tqdm(
                        pool.imap(saver_.exists, df['id'].tolist()),
                        total=len(df),
                        desc='Checking if predictions already exist...',
                    )
                )
            )
        print(f'\nTest samples count: {len(df)}')
        print(f'Predictions already exist for {already_exist_mask.sum()} samples')
        df = df[~already_exist_mask].reset_index(drop=True)
        print(f'Predicting on {len(df)} samples\n')
        return df

    # ##################################### to override:

    def _make_train_transform(self):
        raise NotImplementedError

    def _make_valid_transform(self):
        raise NotImplementedError

    def _make_test_transform(self):
        raise NotImplementedError
