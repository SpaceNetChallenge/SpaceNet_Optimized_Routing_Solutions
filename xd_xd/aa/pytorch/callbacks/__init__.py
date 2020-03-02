import os
from copy import deepcopy

from tensorboardX import SummaryWriter
import torch

from aa.pytorch.callbacks.base import Callback


class EarlyStopper(Callback):
    def __init__(self, patience):
        super().__init__()
        self.patience = patience

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['tot_loss'])
        if loss < self.metrics_collection.best_loss:
            self.metrics_collection.best_loss = loss
            self.metrics_collection.best_epoch = epoch
        if epoch - self.metrics_collection.best_epoch >= self.patience:
            self.metrics_collection.stop_training = True


class Callbacks(Callback):
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            callbacks = callbacks.callbacks
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = []

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_batch_begin(self, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch):
        for callback in self.callbacks:
            callback.on_batch_end(batch)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class ModelSaver(Callback):
    def __init__(self, save_every, save_name, best_only=True):
        super().__init__()
        self.save_every = save_every
        self.save_name = save_name
        self.best_only = best_only

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['tot_loss'])
        need_save = not self.best_only
        if epoch % self.save_every == 0:
            if loss < self.metrics_collection.best_loss:
                self.metrics_collection.best_loss = loss
                self.metrics_collection.best_epoch = epoch
                need_save = True

            if need_save:
                torch.save(deepcopy(self.estimator.model),
                           os.path.join(self.estimator.save_path, self.save_name)
                           .format(epoch=epoch, loss="{:.2}".format(loss)))


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, path):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model_state_dict,
        'optimizer': optimizer_state_dict,
    }, path)


class CheckpointSaver(Callback):
    def __init__(self, save_every, save_name):
        super().__init__()
        self.save_every = save_every
        self.save_name = save_name

    def on_epoch_end(self, epoch):
        loss = float(self.metrics_collection.val_metrics['tot_loss'])
        if epoch % self.save_every == 0:
            save_checkpoint(epoch,
                            self.estimator.model.state_dict(),
                            self.estimator.optimizer.state_dict(),
                            os.path.join(
                                self.estimator.save_path,
                                self.save_name
                            ).format(
                                epoch=epoch,
                                loss="{:.5}".format(loss)))


class TensorBoard(Callback):
    def __init__(self, logdir):
        super().__init__()
        self.logdir = logdir
        self.writer = None

    def on_train_begin(self):
        os.makedirs(self.logdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

    def on_epoch_end(self, epoch):
        for k, v in self.metrics_collection.train_metrics.items():
            self.writer.add_scalar('train/{}'.format(k), float(v), global_step=epoch)

        for k, v in self.metrics_collection.val_metrics.items():
            self.writer.add_scalar('val/{}'.format(k), float(v), global_step=epoch)

        for idx, param_group in enumerate(self.estimator.optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=epoch)

    def on_train_end(self):
        self.writer.close()
