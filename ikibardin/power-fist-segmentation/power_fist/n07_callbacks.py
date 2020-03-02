# Based on https://github.com/selimsef/dsb2018_topcoders/blob/master/albu/src/pytorch_utils/callbacks.py
import logging
import os
from queue import PriorityQueue

import cv2
import torch
from tensorboardX import SummaryWriter


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.runner = None
        self.metrics = None

    def set_trainer(self, runner):
        self.runner = runner
        self.metrics = runner.metrics

    def on_batch_begin(self, i, **kwargs):
        pass

    def on_batch_end(self, i, **kwargs):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_stage_begin(self, stage):
        pass

    def on_stage_end(self, stage):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Callbacks(Callback):
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            callbacks = callbacks.callbacks
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = []

    def set_trainer(self, runner):
        for callback in self.callbacks:
            callback.set_trainer(runner)

    def on_batch_begin(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(i, **kwargs)

    def on_batch_end(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(i, **kwargs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_stage_begin(self, stage):
        for callback in self.callbacks:
            callback.on_stage_begin(stage)

    def on_stage_end(self, stage):
        for callback in self.callbacks:
            callback.on_stage_end(stage)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, save_optim, path):
    if save_optim:
        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': model_state_dict,
                'optimizer': optimizer_state_dict,
            },
            path,
        )
    else:
        torch.save({'epoch': epoch + 1, 'state_dict': model_state_dict}, path)


class CheckpointSaver(Callback):
    def __init__(
        self,
        save_name,
        num_checkpoints,
        metric_name,
        mode,
        save_optim=False,
        local_rank=0,
    ):
        super().__init__()
        self.save_name = save_name
        self._best_checkpoints_queue = PriorityQueue(num_checkpoints)
        self._metric_name = metric_name
        self._mode = mode
        self._save_optim = save_optim
        self._local_rank = local_rank
        assert mode in ('min', 'max')

    def on_stage_begin(self, stage):
        os.makedirs(self._get_model_dir(), exist_ok=True)
        while not self._best_checkpoints_queue.empty():
            self._best_checkpoints_queue.get()

    def on_epoch_end(self, epoch):
        metric = self.metrics.val_metrics[self._metric_name]
        new_path_to_save = os.path.join(
            self._get_model_dir(),
            self.save_name.format(epoch=epoch, metric='{:.5}'.format(metric)),
        )
        improvement = self._try_update_best_losses(metric, new_path_to_save)
        if self._local_rank == 0 and improvement:
            save_checkpoint(
                epoch=epoch,
                model_state_dict=self.runner.model.state_dict(),
                optimizer_state_dict=self.runner.optimizer.state_dict(),
                save_optim=self._save_optim,
                path=new_path_to_save,
            )

    def _get_model_dir(self):
        path = os.path.join(
            self.runner.paths_config['dumps']['path'],
            self.runner.config['train_params']['name'],
            'fold_{}'.format(self.runner.config['data_params']['fold']),
            self.runner.current_stage,
            self.runner.paths_config['dumps']['weights'],
        )

        return path

    def _try_update_best_losses(self, metric, new_path_to_save):
        if self._mode == 'min':
            metric = -metric
        if not self._best_checkpoints_queue.full():
            self._best_checkpoints_queue.put((metric, new_path_to_save))
            return True

        min_metric, min_metric_path = self._best_checkpoints_queue.get()

        if min_metric <= metric:
            if os.path.exists(min_metric_path):
                os.remove(min_metric_path)
            self._best_checkpoints_queue.put((metric, new_path_to_save))
            return True

        self._best_checkpoints_queue.put((min_metric, min_metric_path))
        return False


class TensorBoard(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        self._global_step = 1

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch):
        for k, v in self.metrics.train_metrics.items():
            self.writer.add_scalar('train/{}'.format(k), float(v), global_step=self._global_step)

        for k, v in self.metrics.val_metrics.items():
            self.writer.add_scalar('val/{}'.format(k), float(v), global_step=self._global_step)

        for idx, param_group in enumerate(self.runner.optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=self._global_step)

        self._global_step += 1

    def on_train_end(self):
        self.writer.close()


class Logger(Callback):
    def __init__(self, log_dir, local_rank=0):
        super().__init__()
        self._local_rank = local_rank
        os.makedirs(log_dir, exist_ok=True)
        log_filepath = os.path.join(log_dir, 'logs.txt')
        self.logger = self._get_logger(log_filepath)

    def on_epoch_begin(self, epoch):
        self.logger.info(
            'Stage "{}" | Epoch {} | optimizer "{}" | lr {}'.format(
                self.runner.current_stage,
                epoch,
                self.runner.optimizer.__class__.__name__,
                self._get_current_lr(),
            )
        )

    def on_epoch_end(self, epoch):
        self.logger.info('Train metrics: ' + self._get_metrics_string(self.metrics.train_metrics))
        self.logger.info('Valid metrics: ' + self._get_metrics_string(self.metrics.val_metrics) + '\n')

    def on_stage_begin(self, stage):
        self.logger.info('Starting stage "{}" with params:\n{}\n'.format(stage, self.runner.config['stages'][stage]))

    def on_train_begin(self):
        self.logger.info('Starting training with params:\n{}\n\n'.format(self.runner.config))

    @staticmethod
    def _get_logger(log_filepath):
        logger = logging.getLogger(log_filepath)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_filepath)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _get_current_lr(self):
        res = []
        for param_group in self.runner.optimizer.param_groups:
            res.append(param_group['lr'])
        if len(res) == 1:
            return res[0]
        return res

    def _get_metrics_string(self, metrics):
        return ' | '.join('{}: {:.5f}'.format(k, v) for k, v in metrics.items())


class EarlyStopping(Callback):
    def __init__(self, metric_name, mode='max'):
        super().__init__()
        self._metric_name = metric_name
        if mode not in ('min', 'max'):
            raise ValueError('Unknown mode "{}"'.format(mode))
        self._mode = mode
        self._best_metric = None
        self._metric_not_updated_for = None
        self._current_patience = None

    def on_stage_begin(self, stage):
        self._best_metric = float('-inf') if self._mode == 'max' else float('+inf')
        self._metric_not_updated_for = 0

        stage_config = self.runner.config['stages'][stage]
        if 'early_stopping' not in stage_config:
            self._current_patience = float('+inf')
        else:
            self._current_patience = stage_config['early_stopping']['patience']

    def on_epoch_end(self, epoch):
        self._make_counter_step()
        if self._metric_not_updated_for > self._current_patience:
            self.runner.end_stage()

    def _try_update_best_metric(self):
        metric = self.metrics.val_metrics[self._metric_name]
        if (self._mode == 'max' and
            metric > self._best_metric) or (self._mode == 'min' and metric < self._best_metric):
            self._best_metric = metric
            return True
        return False

    def _make_counter_step(self):
        metric_updated = self._try_update_best_metric()
        if metric_updated:
            self._metric_not_updated_for = 0
        else:
            self._metric_not_updated_for += 1
