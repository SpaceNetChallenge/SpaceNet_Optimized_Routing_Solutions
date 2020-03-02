import glob
import os
from typing import Tuple, Dict

import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    MultiStepLR,
)
from torch.utils.data import Dataset, DataLoader

from power_fist import n07_callbacks as callbacks
from power_fist import metrics
from power_fist.models import models
from power_fist.n03_loss import (
    FocalLoss,
    IoU,
    LossMixture,
    LossBinaryDice,
    IoU_0,
    IoU_1,
    IoU_2,
    IoU_3,
    IoU_4,
    IoU_5,
    IoU_6,
    IoU_7,
    BinaryDice2ch,
    WeightedMultichannelMixture,
    DeepSupervisionLoss,
    TopologyAwareLoss,
)
from power_fist.n10_hard_negative_mining import HardNegativeMiner
from power_fist import n11_lovasz_losses
from power_fist.n12_optimizers import SleepingAdam
from power_fist.n15_symm_lovasz_loss import SymmetricLovaszLoss

OPTIMIZERS = {
    'Adam': Adam,
    'SGD': SGD,
    'SleepingAdam': SleepingAdam,
    'RMSprop': RMSprop,
}

LOSSES = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'BCE': nn.BCEWithLogitsLoss,
    'FocalLoss': FocalLoss,
    'LovaszHingeLoss': n11_lovasz_losses.LovaszHingeLoss,
    'LovaszHingeLoss_elu': n11_lovasz_losses.LovaszHingeLoss_elu,
    'LossMixture': LossMixture,
    'SOTALovaszLoss': SymmetricLovaszLoss,
    'BinaryDice': LossBinaryDice,
    'BinaryDice2ch': BinaryDice2ch,
    'LovaszSigmoid': n11_lovasz_losses.LovaszSigmoid,
    'WeightedMultichannelMixture': WeightedMultichannelMixture,
    'DeepSupLoss': DeepSupervisionLoss,
    'LovaszSoftmaxLoss': n11_lovasz_losses.LovaszSoftmaxLoss,
}

SCHEDULERS = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'StepLR': StepLR,
    'MultiStepLR': MultiStepLR,
    'CosineAnnealingLR': CosineAnnealingLR,
    'ReduceLROnPlateau': ReduceLROnPlateau,
}

METRICS = {
    'Accuracy': metrics.classification.Accuracy,
    'BCE': nn.BCEWithLogitsLoss,
    'DiceScore': metrics.segmentation.DiceScore,
    'IoU': IoU,
    'IoU_0': IoU_0,
    'IoU_1': IoU_1,
    'IoU_2': IoU_2,
    'IoU_3': IoU_3,
    'IoU_4': IoU_4,
    'IoU_5': IoU_5,
    'IoU_6': IoU_6,
    'IoU_7': IoU_7,
    'F2Score': metrics.classification.F2Score,
    'TopologyAwareLoss': TopologyAwareLoss,
}

DATASETS = {}


class UnitsFactory:
    def __init__(self, config: Dict, paths_config: Dict, data_interface_class):
        self._paths = paths_config
        self._config = config
        self._train_params = config['train_params']
        self._checkpoint_loader = CheckpointLoader(config, paths_config)
        self._data_interface_class = data_interface_class

    def make_model(self):
        model_name = self._train_params['model_name']
        num_classes = self._train_params['num_classes']
        model = models[model_name](num_classes=num_classes, pretrained=True)
        if 'weights' not in self._train_params or self._train_params['weights'] is None:
            return model
        raise NotImplementedError

    def make_optimizer(self, model, stage, world_size, batch_size, wtf_lr=False):
        stage_config = self.get_stage_config(stage)
        for p in model.parameters():
            p.requires_grad = True
        if 'freeze_encoder' in stage_config and stage_config['freeze_encoder']:
            if hasattr(model, 'encoder_stages'):
                for p in model.encoder_stages.parameters():
                    p.requires_grad = False
            elif hasattr(model, 'encoder'):
                for p in model.encoder.parameters():
                    p.requires_grad = False
            elif hasattr(model, 'module') and hasattr(model.module, 'encoder_stages'):
                for p in model.module.encoder_stages.parameters():
                    p.requires_grad = False
            elif hasattr(model, 'module') and hasattr(model.module, 'encoder'):
                for p in model.module.encoder.parameters():
                    p.requires_grad = False
            else:
                raise AttributeError('No encoder or encoder_stages, idk what to freeze')
            # if (not hasattr(model, 'encoder_stages')) and (not hasattr(model, 'encoder')):
            #     raise ValueError('idk what to freeze, lol')
        if wtf_lr:
            print('so lr fucked up now')
            stage_config['optimizer_params']['lr'] = (
                    stage_config['optimizer_params']['lr'] * float(batch_size * world_size) / 256.0
            )

        optimizer = OPTIMIZERS[stage_config['optimizer']](
            params=filter(lambda p: p.requires_grad, model.parameters()), **stage_config['optimizer_params']
        )
        return optimizer

    def make_scheduler(self, optimizer, stage):
        stage_config = self.get_stage_config(stage)
        return SCHEDULERS[stage_config['scheduler']](optimizer=optimizer, **stage_config['scheduler_params'])

    def make_loss(self) -> nn.Module:
        if 'loss_params' not in self._train_params:
            return LOSSES[self._train_params['loss']]()
        return LOSSES[self._train_params['loss']](**self._train_params['loss_params'])

    # noinspection PyTypeChecker
    def make_callbacks(self, local_rank) -> callbacks.Callbacks:
        # TODO: ugly
        log_dir = os.path.join(self.get_experiment_dir(), self._paths['dumps']['logs'])
        if local_rank == 0:
            callbacks_list = [callbacks.Logger(log_dir), callbacks.TensorBoard(log_dir)]
        else:
            callbacks_list = []
        if 'early_stopping' in self._train_params:
            callbacks_list.append(callbacks.EarlyStopping(**self._train_params['early_stopping']))

        if 'checkpoint_saver' in self._train_params:
            callbacks_list.append(
                callbacks.CheckpointSaver(
                    save_name='epoch{epoch}_metric{metric}.pth',
                    local_rank=local_rank,
                    **self._train_params['checkpoint_saver']
                )
            )

        return callbacks.Callbacks(callbacks_list)

    def make_train_valid_datasets(self, stage: str) -> Tuple[Dataset, Dataset]:
        return self._get_data_interface(stage=stage).make_train_valid_datasets()

    def make_train_valid_loaders(self, stage: str, distributed=False) -> Tuple[DataLoader, DataLoader]:
        return self._get_data_interface(stage).make_train_valid_loaders(distributed=distributed)

    def make_test_loader(self, part, saver_):
        data_interface = self._get_data_interface(stage='predict')
        test_loader = data_interface.make_test_loader(part, saver_)
        return test_loader

    def make_batch_handler(self, stage):
        data_interface = self._get_data_interface(stage)
        training = stage != 'predict'
        return data_interface.make_batch_handler(training)

    def make_predictions_saver(self, checkpoint_path, dataset_part):
        data_interface = self._get_data_interface(stage='predict')
        return data_interface.make_predictions_saver(self._train_params['name'], checkpoint_path, dataset_part)

    def get_best_checkpoints_paths(self, num_checkpoints=None):
        return self._checkpoint_loader.get_best_checkpoints_paths(num_checkpoints=num_checkpoints)

    def get_best_checkpoint_from_stage(self, stage):
        return self._checkpoint_loader.get_best_checkpoint_from_stage(stage)

    def make_metrics(self):
        metrics = dict()
        for metric_name in self._train_params['metrics']:
            if 'Salt' in metric_name:
                metrics[metric_name] = METRICS[metric_name](crop_size=self._config['data_params']['upscale_size'])
            else:
                metrics[metric_name] = METRICS[metric_name]()
        return metrics

    def make_hard_negative_miner(self):
        if 'negative_mining' not in self._train_params:
            return None
        return HardNegativeMiner(rate=self._train_params['negative_mining']['rate'])

    #######################################################################################################

    def _get_data_interface(self, stage):
        data_params = self._config['data_params'].copy()

        if stage != 'predict':
            stage_config = self.get_stage_config(stage)
            if 'data_params' in stage_config:
                local_data_params = stage_config['data_params']
                data_params.update(local_data_params)

        return self._data_interface_class(self._paths, data_params, self._config['predict_params'])

    def get_stage_config(self, stage):
        return self._config['stages'][stage]

    def get_experiment_dir(self) -> str:
        return self._checkpoint_loader.get_experiment_dir()


class CheckpointLoader:
    def __init__(self, config, paths_config):
        self._paths = paths_config
        self._config = config

    def get_best_checkpoints_paths(self, num_checkpoints=None):
        all_checkpoints = self._get_all_checkpoints()
        if len(all_checkpoints) == 0:
            raise ValueError('No checkpoints found')

        if num_checkpoints is None:
            num_checkpoints = num_checkpoints
        chosen_checkpoints = self._choose_n_best(all_checkpoints, num_checkpoints)
        print('CHOSEN CHECKPOINTS: ', chosen_checkpoints)
        return chosen_checkpoints

    def get_best_checkpoint_from_stage(self, stage):
        stage_checkpoints = self._get_stage_checkpoints(stage)
        return self._choose_n_best(stage_checkpoints, 1)[0]

    def _get_all_checkpoints(self):
        all_checkpoints = []
        for stage in self._config['stages']:
            all_checkpoints += self._get_stage_checkpoints(stage)
        return np.array(all_checkpoints)

    @staticmethod
    def _get_checkpoint_metric(checkpoint):
        return float(checkpoint.split('metric')[1].split('.pth')[0])

    def get_experiment_dir(self):
        return os.path.join(
            self._paths['dumps']['path'],
            self._config['train_params']['name'],
            'fold_{}'.format(self._config['data_params']['fold']),
        )

    def _get_stage_checkpoints(self, stage):
        # print(self._get_weights_dir(stage))
        return glob.glob(os.path.join(self._get_weights_dir(stage), '*.pth'))

    def _get_weights_dir(self, stage):
        return os.path.join(self.get_experiment_dir(), stage, self._paths['dumps']['weights'])

    def _choose_n_best(self, checkpoints, num_to_choose):
        checkpoints = np.array(checkpoints)
        metrics = [self._get_checkpoint_metric(ch) for ch in checkpoints]
        if self._config['predict_params']['metric_mode'] == 'max':
            metrics = [-m for m in metrics]
        best_indices = np.array(np.argsort(metrics)[:num_to_choose], dtype=int)
        return checkpoints[best_indices]
