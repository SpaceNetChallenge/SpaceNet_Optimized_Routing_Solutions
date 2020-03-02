import os
from collections import defaultdict
from typing import Dict

import apex
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from power_fist.n06_units_factory import UnitsFactory


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def mixup_data(x, y, alpha=0.2, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def fix_checkpoint(best_checkpoint_path, model_state_dict):
    loaded_state_dict = torch.load(best_checkpoint_path, map_location='cpu')['state_dict']

    sanitized = dict()
    for key in loaded_state_dict.keys():
        if key.startswith('module.'):
            sanitized[key[7:]] = loaded_state_dict[key]
        else:
            sanitized[key] = loaded_state_dict[key]

    # for key in ["final.0.weight", "final.0.bias"]:
    #     if key in model_state_dict.keys() or key in sanitized.keys():
    #         if model_state_dict[key].size() != sanitized[key].size():
    #             print(f"Skipping key {key} due to size mismatch")
    #             sanitized.pop(key, None)

    return sanitized


class Metrics:
    def __init__(self):
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class Runner:
    def __init__(self, config: Dict, paths_config: Dict, data_interface_class, local_rank=0):
        self._debugger = Debugger(
            config=config,
            paths_config=paths_config,
            data_interface_class=data_interface_class,
        )
        self._trainer = Trainer(
            config=config,
            paths_config=paths_config,
            data_interface_class=data_interface_class,
            local_rank=local_rank,
        )
        self._predictor = Predictor(
            config=config,
            paths_config=paths_config,
            data_interface_class=data_interface_class,
            local_rank=local_rank,
        )

    def run_benchmark(self):
        self._debugger.run_benchmark()

    def test_loader(self):
        self._debugger.test_loader()

    def run_training(self):
        self._trainer.run_training()

    def predict(self):
        self._predictor.run()

    def resume_training(self):
        self._trainer.resume_training()


class Debugger:
    def __init__(self, config: Dict, paths_config: Dict, data_interface_class):
        self._config = config
        self._paths_config = paths_config

        self._units_factory = UnitsFactory(
            config=config,
            paths_config=paths_config,
            data_interface_class=data_interface_class,
        )

        self._output_dir = os.path.join(
            self._units_factory.get_experiment_dir(),
            self._paths_config['dumps']['samples'],
        )

        self._IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
        self._IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    def run_benchmark(self):
        model = self._units_factory.make_model()
        input_ = torch.randn(1, 3, 832, 1344)

        if torch.cuda.is_available():
            model = nn.DataParallel(model).cuda()
            input_ = input_.cuda()

        run_speed_test(model, input_)
        del model
        del input_
        torch.cuda.empty_cache()

    def test_loader(self):
        for stage in self._config['stages']:
            for dataset in self._units_factory.make_train_valid_datasets(stage=stage):

                if dataset.get_mode() != 'train':
                    continue  # FIXME

                dumping_dir = os.path.join(self._output_dir, stage, dataset.get_mode())
                os.makedirs(dumping_dir, exist_ok=True)

                for index in tqdm(
                        range(0, min(100, len(dataset))),
                        desc=f'Dumping {dataset.get_mode()} images for stage `{stage}`'
                ):
                    data = dataset[index]

                    image = self._inverse_post_transform(data['image'], mode='image')
                    mask = self._inverse_post_transform(data['target'], mode='mask')

                    visualized = cv2.cvtColor(
                        self._apply_mask(image, mask),
                        cv2.COLOR_RGB2BGR,
                    )

                    cv2.imwrite(os.path.join(dumping_dir, f'{data["id"]}.jpg'), visualized)

    # FIXME: probably task-specific, move somewhere else
    def _inverse_post_transform(self, image: torch.Tensor, mode: str) -> np.ndarray:
        if mode == 'image':
            image = image.cpu().numpy()
            if len(image.shape) != 3:
                raise ValueError(f'Expected image tensor to have (C, H, W) shape, got {image.shape}')
            image = np.transpose(image, (1, 2, 0))
            image = self._IMAGENET_MEAN + self._IMAGENET_STD * image
            return (image * 255.).astype(np.uint8)

        if mode == 'mask':
            image = image.cpu().numpy()

            if len(image.shape) == 3:
                # print(image.shape)
                image = np.transpose(image, (1, 2, 0))[:, :, 0]
                print('pipeline', image.max())
            elif image.shape != 2:
                raise ValueError(f'Expected image tensor to have (C, H, W) or (H, W) shape, got {image.shape}')
            return image

        raise ValueError(f'Expected mode to be either `image` or `mask`, got `{mode}`')

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if len(mask.shape) > 2:
            raise NotImplementedError(f'Currently implemented only for (H, W) masks')
        cnv = image.copy()
        overlay = cnv.copy()
        overlay[mask > 0.5] = (255, 0, 0)
        ALPHA = 0.3
        cv2.addWeighted(overlay, ALPHA, cnv, 1 - ALPHA, 1, cnv)
        return cnv


class Trainer:
    def __init__(self, config, paths_config, data_interface_class, local_rank=0):
        self._units_factory = UnitsFactory(config, paths_config, data_interface_class)

        self.config = config
        self.paths_config = paths_config
        self.model = None
        self.loss = self._units_factory.make_loss().cuda()

        self.metrics = Metrics()
        self.current_stage = None
        self.optimizer = None
        self.scheduler = None
        self._train_loader = None
        self._valid_loader = None
        self._valid2_loader = None
        self._batch_handler = None

        self._metric_funcs = self._units_factory.make_metrics()
        self._negative_miner = self._units_factory.make_hard_negative_miner()
        self._stage_running = False

        self._distributed = False
        if 'WORLD_SIZE' in os.environ:
            self._distributed = int(os.environ['WORLD_SIZE']) > 1

        self._world_size = 1
        self._local_rank = local_rank

        if self._distributed:
            print(self._local_rank)
            torch.cuda.set_device(self._local_rank)
            torch.distributed.init_process_group(
                backend='nccl',
                # init_method='env://',
                rank=local_rank,
            )

            self._world_size = torch.distributed.get_world_size()

        self._callbacks = self._units_factory.make_callbacks(local_rank=self._local_rank)
        self._callbacks.set_trainer(self)

        self._mixup = False
        if 'mixup' in self.config['data_params']:
            self._mixup = self.config['data_params']['mixup']
            print(f"add mixup augs {self.config['data_params']['mixup']}")

        self._accum = False
        self._counter = 0
        if 'accum' in self.config['data_params']:
            self._accum = self.config['data_params']['accum']
            print(f"add barth accumulation {self.config['data_params']['accum']}")

        self._warmup = False
        if 'warmup' in self.config['data_params']:
            self._warmup = self.config['data_params']['warmup']
            print(f"set warmup {self.config['data_params']['warmup']}")

        self._val2 = False
        if 'val2' in self.config['data_params']:
            self._val2 = self.config['data_params']['val2']
            print(f"set val2 {self.config['data_params']['val2']}")

    def resume_training(self):
        self.run_training(resume=True)

    def run_training(self, resume=False):
        self._callbacks.on_train_begin()
        for stage in self.config['stages']:
            self.model = self._setup_model(resume=resume)
            self.current_stage = stage
            self._train_loader, self._valid_loader = self._units_factory.make_train_valid_loaders(
                stage,
                distributed=self._distributed,
            )
            self._batch_handler = self._units_factory.make_batch_handler(stage)
            self.optimizer = self._units_factory.make_optimizer(
                self.model,
                stage,
                self._world_size,
                self.config['data_params']['batch_size'],
                wtf_lr=self._distributed,
            )

            if 'apex' in self.config.keys():
                self.model, self.optimizer = amp.initialize(
                    self.model,
                    self.optimizer,
                    opt_level=self.config['apex']['opt_level'],
                    keep_batchnorm_fp32=self.config['apex']['keep_batchnorm_fp32'],
                    loss_scale=self.config['apex']['loss_scale'],
                )

                if self._distributed:
                    self.model = DDP(self.model, delay_allreduce=True)
                    print('DDP')
            else:
                print('no apex')
            self.scheduler = self._units_factory.make_scheduler(self.optimizer, stage)

            self._run_one_stage(stage)
        self._callbacks.on_train_end()

    def end_stage(self):
        self._stage_running = False

    def _setup_model(self, resume=False):
        model = self._units_factory.make_model()
        if resume:
            best_checkpoint_path = self._units_factory.get_best_checkpoints_paths(num_checkpoints=1)
            # if best_checkpoint_path is not None:
            best_checkpoint_path = best_checkpoint_path[0]
            model_state_dict = model.state_dict()

            sanitized = fix_checkpoint(best_checkpoint_path, model_state_dict)

            model_state_dict.update(sanitized)
            model.load_state_dict(model_state_dict, strict=False)
            print('checkpoint loaded')

        if 'apex' in self.config.keys():
            if 'sync_bn' in self.config['train_params'].keys():
                if self.config['train_params']['sync_bn']:
                    model = apex.parallel.convert_syncbn_model(model)
                    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = model.cuda()
            # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            model = nn.DataParallel(model).cuda()

        # model = SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def _run_one_stage(self, stage):
        self._counter = 0
        self._stage_running = True
        self._callbacks.on_stage_begin(stage)
        for epoch in range(self._units_factory.get_stage_config(stage)['epochs']):
            self._callbacks.on_epoch_begin(epoch)

            self.model.train()
            self.metrics.train_metrics = self._run_one_epoch(epoch, self._train_loader, is_train=True)

            self.model.eval()
            self.metrics.val_metrics = self._run_one_epoch(epoch, self._valid_loader, is_train=False)

            if self._val2:
                self.model.eval()
                val2_report = self._run_one_epoch(epoch, self._valid2_loader, is_train=False)
                for key, value in val2_report.items():
                    self.metrics.val_metrics[f't_{key}'] = value

            if self._local_rank == 0:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(self.metrics.val_metrics['loss'], epoch)
                else:
                    self.scheduler.step(epoch)

            # if self._local_rank == 0:
            self._callbacks.on_epoch_end(epoch)
            if not self._stage_running:
                break

        best_checkpoint_path = self._units_factory.get_best_checkpoint_from_stage(stage)
        model_state_dict = self.model.state_dict()

        sanitized = fix_checkpoint(best_checkpoint_path, model_state_dict)

        model_state_dict.update(sanitized)
        self.model.load_state_dict(model_state_dict, strict=False)

        print('\nLoading best checkpoint from stage "{}": "{}"\n'.format(stage, best_checkpoint_path))

        self._callbacks.on_stage_end(stage)
        torch.cuda.empty_cache()

    def _get_progress_bar(self, epoch, loader, is_train):
        generator = enumerate(loader)
        if self._local_rank == 0:
            if not is_train:
                return tqdm(
                    generator,
                    total=len(loader),
                    desc=f'Epoch {epoch}  validating...',
                    ncols=0,
                )
            else:
                return tqdm(
                    generator,
                    total=self.config['data_params']['steps_per_epoch'],
                    desc=f'Epoch {epoch}  training...  ',
                    ncols=0,
                )
        else:
            return generator

    def _run_one_epoch(self, epoch, loader, is_train=True):
        epoch_report = defaultdict(float)

        progress_bar = self._get_progress_bar(epoch, loader, is_train)
        with torch.set_grad_enabled(is_train):
            for i, data in progress_bar:
                if epoch < 1 and self._warmup is True:
                    self.adjust_learning_rate(
                        self.config['stages'][self.current_stage]['optimizer_params']['lr'],
                        self.config['data_params']['steps_per_epoch'],
                    )

                step_report = self._make_step(i, data, is_train)

                # if self._local_rank == 0:
                for key, value in step_report.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    epoch_report[key] += value

                if is_train and self._local_rank == 0:
                    progress_bar.set_postfix(**{k: '{:.5f}'.format(v / (i + 1)) for k, v in epoch_report.items()})

                    if self._negative_miner is not None:
                        self._negative_miner.update_cache(step_report, data)
                        if self._negative_miner.need_iter():
                            self._make_step(i, self._negative_miner.cache, is_train)
                            self._negative_miner.invalidate_cache()

                if is_train and i >= self.config['data_params']['steps_per_epoch']:
                    break
        return {key: value / len(loader) for key, value in epoch_report.items()}

    def _make_step(self, batch_idx, data, is_train):
        self._callbacks.on_batch_begin(batch_idx)
        report = {}

        if is_train:
            if self._accum:
                if self._counter % self._accum == 0:
                    self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()

        self._counter += 1

        images, labels = self._batch_handler(data)
        if self._mixup and is_train:
            images, labels_a, labels_b, lam = mixup_data(images, labels, use_cuda=True)
        assert not torch.isnan(images).any()

        predictions = self.model(images)
        assert not torch.isnan(predictions).any()
        # print(f'\n\n >>>>>> LABELS SIZE {labels.size()}   PREDS SIZE {predictions.size()}  \n\n')
        if self._mixup and is_train:
            loss = mixup_criterion(self.loss, predictions, labels_a, labels_b, lam)
        else:
            loss = self.loss(predictions, labels)

        if self._distributed:
            reduced_loss = reduce_tensor(loss.data, self._world_size)
        else:
            reduced_loss = loss

        assert not torch.isnan(loss)

        report['loss'] = reduced_loss.data
        for metric_name, func in self._metric_funcs.items():
            if is_train and 'Airbus' in metric_name:
                continue
            if self._distributed:
                reduced_func = reduce_tensor(func(predictions, labels), self._world_size)
            else:
                reduced_func = func(predictions, labels)

            report[metric_name] = reduced_func

        if is_train:
            if 'apex' in self.config.keys():
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # FIXME
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            # report['grad'] = grad_norm

            if self._accum:
                if self._counter % self._accum == 0:
                    self.optimizer.step()
            else:
                self.optimizer.step()

        self._callbacks.on_batch_end(
            batch_idx,
            step_report=report,
            is_train=is_train,
            batch_data=data,
            batch_preds=predictions,
        )
        return report

    def adjust_learning_rate(self, lr, len_epoch):
        """Warmup"""
        if self._counter < len_epoch - 1:
            lr = lr * float(9 * self._counter + len_epoch) / (10 * len_epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class Predictor:
    def __init__(self, config, paths_config, data_interface_class, local_rank: int):
        self._units_factory = UnitsFactory(config, paths_config, data_interface_class)
        self._parts_to_predict = config['predict_params']['part']
        self._num_checkpoints = config['predict_params']['num_checkpoints']

        self._batch_handler = self._units_factory.make_batch_handler(stage='predict')

        self._distributed = False
        if 'WORLD_SIZE' in os.environ:
            self._distributed = int(os.environ['WORLD_SIZE']) > 1

        self._world_size = 1
        self._local_rank = local_rank

        # if self._distributed:
        #     torch.cuda.set_device(self._local_rank)
        #     torch.distributed.init_process_group(
        #         backend='nccl',
        #         init_method='env://',
        #     )
        #     self._world_size = torch.distributed.get_world_size()

    def run(self):
        best_checkpoints = self._units_factory.get_best_checkpoints_paths()
        for checkpoint_path in best_checkpoints[:self._num_checkpoints]:
            self._predict_checkpoint(checkpoint_path)

    def _predict_checkpoint(self, checkpoint_path):
        model = self._units_factory.make_model()
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        model.load_state_dict(self._remove_prefix_from_keys(state_dict, prefix='module.'), strict=True)  # FIXME
        model = nn.DataParallel(model).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        for part in self._parts_to_predict:
            self._predict_on_part(checkpoint_path, model, part)

    def _predict_on_part(self, checkpoint_path, model, part):
        predictions_saver = self._units_factory.make_predictions_saver(checkpoint_path, part)

        loader = self._units_factory.make_test_loader(part, predictions_saver)

        progress_bar = tqdm(enumerate(loader), total=len(loader), desc='Predicting', ncols=0)

        with torch.set_grad_enabled(False):
            for i, data in progress_bar:
                self._make_step(model, predictions_saver, data)

        predictions_saver.save()

    def _make_step(self, model, predictions_saver, data):
        images, ids = self._batch_handler(data)
        predictions = model(images)
        predictions_saver.add(ids, predictions)

    @staticmethod
    def _remove_prefix_from_keys(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        new_state_dict = dict()
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_state_dict[key[len(prefix):]] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
