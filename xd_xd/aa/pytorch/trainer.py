from pathlib import Path
from logging import getLogger
import importlib
import os
import time
from collections import defaultdict
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
from tqdm import tqdm


from aa.pytorch.dataset_sp5r2 import TrainDataset, ValDataset
from aa.pytorch.loss import (
    dice,
    focal,
    focal_cannab,
    soft_dice_loss,
    weight_reshape,
)
from aa.pytorch.callbacks import (
    EarlyStopper,
    ModelSaver,
    TensorBoard,
    CheckpointSaver,
    Callbacks,
)
import aa.cli.sp5r2.util as u


torch.backends.cudnn.benchmark = True


logger = getLogger('aa')


class Estimator(object):
    def __init__(self, model: torch.nn.Module, optimizer: Type[optim.Optimizer], save_path, config):
        # self.model = nn.DataParallel(model).cuda()
        self.model = model.cuda()
        self.optimizer = optimizer(self.model.parameters(), lr=config.lr)

        self.start_epoch = 0
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.iter_size = config.iter_size

        self.lr_scheduler = None
        self.lr = config.lr
        self.config = config
        self.optimizer_type = optimizer

    def resume(self, checkpoint_name):
        try:
            checkpoint = torch.load(os.path.join(self.save_path, checkpoint_name))
        except FileNotFoundError:
            print("Attempt to resume failed, file not found")
            print ("  Missing file:", os.path.join(self.save_path, checkpoint_name))
            return False

        self.start_epoch = checkpoint['epoch']

        model_dict = self.model.state_dict()
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        print("resumed from checkpoint {} on epoch: {}".format(os.path.join(self.save_path, checkpoint_name), self.start_epoch))
        return True

    def calculate_loss_single_channel(self, output, target, meter, training,
                                      iter_size, weight_channel=None):
        if weight_channel:
            output, target = weight_reshape(output, target,
                                            weight_channel=weight_channel,
                                            min_weight_val=0.16)

        bce = F.binary_cross_entropy_with_logits(output, target)
        if 'ce' in self.config.loss.keys():
            pass
        else:
            output = torch.sigmoid(output)

        d = dice(output, target)
        dice_l = 1 - d
        dice_soft_l = soft_dice_loss(output, target)
        focal_l = focal(output, target)

        smooth_l1_mult = 100
        smooth_l1_l = F.smooth_l1_loss(output, target) * smooth_l1_mult

        mse_mult = 10
        mse_l = F.mse_loss(output, target) * mse_mult

        # custom loss function
        if 'focal' in self.config.loss.keys():
            loss = (self.config.loss['focal'] * focal_l + self.config.loss['dice'] * (1 - d) ) / iter_size
        elif 'bce' in self.config.loss.keys():
            loss = (self.config.loss['bce'] * bce + self.config.loss['dice'] * (1 - d)) / iter_size
        elif 'focal_cannab' in self.config.loss.keys():
            focal_l = focal_cannab(output, target)
            loss = (self.config.loss['focal_cannab'] * focal_l + self.config.loss['soft_dice'] * dice_soft_l) / iter_size
        elif 'smooth_l1' in self.config.loss.keys():
            loss = (self.config.loss['smooth_l1'] * smooth_l1_l + self.config.loss['dice'] * (1 - d)) / iter_size
        elif 'mse' in self.config.loss.keys():
            loss = (self.config.loss['mse'] * mse_l + self.config.loss['dice'] * (1 - d)) / iter_size

        if training:
            loss.backward()

        meter['tot_loss'] += loss.data.cpu().numpy()
        meter['focal'] += focal_l.data.cpu().numpy() / iter_size

        meter['dice_loss'] += dice_l.data.cpu().numpy() / iter_size
        meter['smooth_l1'] += smooth_l1_l.data.cpu().numpy() / iter_size
        meter['mse'] += mse_l.data.cpu().numpy() / iter_size

        return meter

    def make_step_itersize(self, images, ytrues, training, epoch, verbose=False):
        iter_size = self.iter_size
        if training:
            self.optimizer.zero_grad()

        inputs = images.chunk(iter_size)
        targets = ytrues.chunk(iter_size)

        meter = defaultdict(float)
        if training:
            for input, target in zip(inputs, targets):
                input = torch.autograd.Variable(input.cuda())
                target = torch.autograd.Variable(target.cuda())
                output = self.model(input)
                meter = self.calculate_loss_single_channel(output,
                                                           target,
                                                           meter,
                                                           training,
                                                           iter_size)
        else:
            with torch.no_grad():
                for input, target in zip(inputs, targets):
                    input = torch.autograd.Variable(input.cuda())
                    target = torch.autograd.Variable(target.cuda())
                    output = self.model(input)
                    meter = self.calculate_loss_single_channel(output,
                                                               target,
                                                               meter,
                                                               training,
                                                               iter_size)

        if training:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)

        return meter, None


class MetricsCollection(object):
    def __init__(self):
        self.stop_training = False
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class PytorchTrain(object):
    """
    fit, run one epoch, make step
    """
    def __init__(self,
                 estimator: Estimator,
                 fold,
                 callbacks=None,
                 no_eval_period=0,
                 conf=None):
        self.conf = conf
        self.fold = fold
        self.estimator = estimator
        self.no_eval_period = no_eval_period
        print('no_eval_period', self.no_eval_period)

        self.devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        if os.name == 'nt':
            self.devices = ','.join(str(d + 5) for d in map(int, self.devices.split(',')))

        self.metrics_collection = MetricsCollection()

        self.estimator.resume("fold" + str(fold) + "_checkpoint.pth")

        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(self)

    def _run_one_epoch(self, epoch, loader, training=True):
        avg_meter = defaultdict(float)

        desc_str = "Fold {}; Epoch {}{}".format(
            self.fold, epoch, ' eval' if not training else ""
        )
        pbar = tqdm(enumerate(loader), total=len(loader), desc=desc_str, ncols=0)

        for i, data in pbar:
            self.callbacks.on_batch_begin(i)
            meter, ypreds = self._make_step(data, training, epoch)
            for k, val in meter.items():
                avg_meter[k] += val

            pbar.set_postfix(**{k: "{:.5f}".format(v / (i + 1)) for k, v in avg_meter.items()})

            self.callbacks.on_batch_end(i)
        return {k: v / len(loader) for k, v in avg_meter.items()}

    def _make_step(self, data, training, epoch):
        images = data['image']
        ytrues = data['mask']

        meter, ypreds = self.estimator.make_step_itersize(images, ytrues, training, epoch)

        return meter, ypreds

    def fit(self, train_loader, val_loader, nb_epoch):
        self.callbacks.on_train_begin()

        t0 = time.time()
        for epoch in range(self.estimator.start_epoch, nb_epoch):
            self.callbacks.on_epoch_begin(epoch)

            self.estimator.model.train()
            self.metrics_collection.train_metrics = self._run_one_epoch(
                epoch, train_loader, training=True)

            if epoch > self.no_eval_period:
                self.estimator.model.eval()
                self.metrics_collection.val_metrics = self._run_one_epoch(
                    epoch, val_loader, training=False)
            else:
                # Skip
                self.metrics_collection.val_metrics['tot_loss'] = float('inf')

            t1 = time.time()
            print ("\nTotal time elapsed:", (t1 - t0)/60., "minutes")

            self.callbacks.on_epoch_end(epoch)

            if self.metrics_collection.stop_training:
                break

        self.callbacks.on_train_end()


def dynamic_load(model_class_fqn):
    module_name = '.'.join(model_class_fqn.split('.')[:-1])
    class_name = model_class_fqn.split('.')[-1]

    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls


def train(ds, fold, train_idx, val_idx, conf,
          val_ds=None,
          transforms=None,
          val_transforms=None):
    if conf.model_fqn.endswith('SeResnext50_32d4d_upsample'):
        model = dynamic_load(conf.model_fqn)(
            num_classes=conf.num_classes,
            num_channels=conf.num_channels,
            pretrained_file=(
                conf.pretrained_model if 'pretrained_model' in conf else None
            ),
        )
    else:
        model = dynamic_load(conf.model_fqn)(
            num_classes=conf.num_classes,
            num_channels=conf.num_channels,
        )
    # save_path = u.prefix_path() + f'/working/sp5r2/models/weights/{conf.modelname}/fold{fold}'
    save_path = f'/wdata/working/sp5r2/models/weights/{conf.modelname}/fold{fold}'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # tfb_path = u.prefix_path() + f'/working/sp5r2/models/logs/{conf.modelname}/fold{fold}'
    tfb_path = f'/wdata/working/sp5r2/models/logs/{conf.modelname}/fold{fold}'
    Path(tfb_path).mkdir(parents=True, exist_ok=True)

    optimizer = dynamic_load(conf.optimizer_fqn)
    estimator = Estimator(model, optimizer, save_path, config=conf)
    estimator.lr_scheduler = MultiStepLR(estimator.optimizer,
                                         conf.lr_steps,
                                         gamma=conf.lr_gamma)
    if 'scheduler' in conf:
        scheduler_class = dynamic_load(conf.scheduler)
        if conf.scheduler.endswith('CosineAnnealingLR'):
            conf.scheduler_params['optimizer'] = estimator.optimizer
        estimator.lr_scheduler = scheduler_class(**conf.scheduler_params)

    callbacks = [
        ModelSaver(1, ("fold"+str(fold)+"_best.pth"), best_only=True),
        ModelSaver(1, ("fold"+str(fold)+"_last.pth"), best_only=False),
        CheckpointSaver(1, ("fold"+str(fold)+"_checkpoint.pth")),
        CheckpointSaver(1, ("fold"+str(fold)+"_ep{epoch}_{loss}_checkpoint.pth")),
        TensorBoard(tfb_path),
    ]
    if 'early_stopper_patience' in conf:
        callbacks.append(EarlyStopper(conf.early_stopper_patience))

    trainer = PytorchTrain(estimator,
                           conf=conf,
                           fold=fold,
                           callbacks=callbacks,
                           no_eval_period=conf.get('no_eval_period', 0))

    train_dataset = TrainDataset(ds, train_idx, conf,
                                 transforms=transforms,
                                 verbose=False)
    train_loader = PytorchDataLoader(train_dataset,
                                     batch_size=conf.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=conf.num_workers,
                                     pin_memory=True)

    val_dataset = ValDataset(val_ds if val_ds is not None else ds,
                             val_idx, conf,
                             transforms=val_transforms)
    val_loader = PytorchDataLoader(val_dataset,
                                   batch_size=conf.batch_size if not conf.ignore_target_size else 1,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=conf.num_workers,
                                   pin_memory=True)

    trainer.fit(train_loader, val_loader, conf.nb_epoch)
