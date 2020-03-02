import argparse
import os
import subprocess
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from vectorize import vectorize_dir
import models

from albumentations import Compose, RandomSizedCrop, HorizontalFlip, VerticalFlip, RGBShift, RandomBrightnessContrast, \
    RandomGamma, OneOf, RandomRotate90, PadIfNeeded, Transpose, RandomCrop, Rotate, ShiftScaleRotate
from torch.nn.functional import softmax

import losses
from dataset.spacenet_dataset import SpacenetSimpleDataset

from apex.parallel import DistributedDataParallel, convert_syncbn_model
from tensorboardX import SummaryWriter

from tools.config import load_config
from tools.utils import create_optimizer

from apex import amp

from losses import dice_round

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_train_transforms(conf):
    height = conf['crop_height']
    width = conf['crop_width']
    return Compose([
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0),
        OneOf([RandomSizedCrop(min_max_height=(int(height * 0.8), int(height * 1.2)), w2h_ratio=1., height=height,
                               width=width, p=0.9),
               RandomCrop(height=height, width=width, p=0.1)], p=1),
        Rotate(limit=10, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        Transpose(),
        OneOf([RGBShift(), RandomBrightnessContrast(), RandomGamma()], p=0.5),
    ])

def create_val_transforms(conf):
    return Compose([
        PadIfNeeded(min_height=1344, min_width=1344),
    ])


def main():
    parser = argparse.ArgumentParser("PyTorch Severstal Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
    arg('--workers', type=int, default=8, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='spacenet_')
    arg('--data-dir', type=str, default="/wdata")
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--logdir', type=str, default='logs')
    arg('--from-zero', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg('--freeze-epochs', type=int, default=1)
    arg("--local_rank", default=0, type=int)
    arg("--predictions", default="/wdata/oof_preds", type=str)
    arg("--truth_csv", default="oof_gt.txt", type=str)
    arg("--test_every", type=int, default=5)
    arg("--visualizer-path", default="visualizer/visualizer.jar", type=str)

    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    conf = load_config(args.config)
    if 'dla' not in conf['network']:
        model = models.__dict__[conf['network']](seg_classes=conf['num_classes'], backbone_arch=conf['encoder'])
    else:
        model = models.__dict__[conf['network']](classes=conf['num_classes'])
    model = model.cuda()
    if args.distributed:
        model = convert_syncbn_model(model)
    speed_loss_function = losses.__dict__[conf["speed_loss"]["type"]](**conf["speed_loss"]["params"]).cuda()
    mask_loss_function = losses.__dict__[conf["mask_loss"]["type"]](**conf["mask_loss"]["params"]).cuda()
    junction_loss_function = losses.__dict__[conf["junction_loss"]["type"]](**conf["junction_loss"]["params"]).cuda()
    loss_functions = {"speed_loss": speed_loss_function, "mask_loss": mask_loss_function, "junction_loss": junction_loss_function}
    optimizer, scheduler = create_optimizer(conf['optimizer'], model)

    dice_best = 0
    apls_best = 0
    start_epoch = 0
    batch_size = conf['optimizer']['batch_size']

    data_train = SpacenetSimpleDataset(mode="train",
                                       fold=args.fold,
                                       image_type=conf["image_type"],
                                       data_path=args.data_dir,
                                       folds_csv=args.folds_csv,
                                       transforms=create_train_transforms(conf['input']),
                                       multiplier=conf["data_multiplier"],
                                       normalize=conf["input"].get("normalize", None))
    data_val = SpacenetSimpleDataset(mode="val",
                                     image_type=conf["image_type"],
                                     fold=args.fold,
                                     data_path=args.data_dir,
                                     folds_csv=args.folds_csv,
                                     transforms=create_val_transforms(conf['input']),
                                     normalize=conf["input"].get("normalize", None)
                                     )
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=args.workers,
                                   shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False,
                                   drop_last=True)
    val_batch_size = 1
    val_data_loader = DataLoader(data_val, batch_size=val_batch_size, num_workers=args.workers, shuffle=False,
                                 pin_memory=False)

    os.makedirs(args.logdir, exist_ok=True)
    summary_writer = SummaryWriter(args.logdir + '/' + args.prefix + conf['encoder'])
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            if conf['optimizer'].get('zero_decoder', False):
                for key in state_dict.copy().keys():
                    if key.startswith("module.final"):
                        del state_dict[key]
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint['epoch']
                dice_best = checkpoint['dice_best']
                apls_best = checkpoint.get('apls_best', 0)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.from_zero:
        start_epoch = 0
    current_epoch = start_epoch

    if conf['fp16']:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O0',
                                          loss_scale='dynamic')

    snapshot_name = "{}{}_{}_{}".format(args.prefix, conf['network'], conf['encoder'], args.fold)

    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)
    else:
        model = DataParallel(model).cuda()
    for epoch in range(start_epoch, conf['optimizer']['schedule']['epochs']):
        if epoch < args.freeze_epochs:
            print("Freezing encoder!!!")
            model.module.encoder_stages.eval()
            for p in model.module.encoder_stages.parameters():
                p.requires_grad = False
        else:
            print("Unfreezing encoder!!!")
            model.module.encoder_stages.train()
            for p in model.module.encoder_stages.parameters():
                p.requires_grad = True
        train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                    args.local_rank)

        model = model.eval()
        if args.local_rank == 0:
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'dice_best': dice_best,
                'apls_best': apls_best,
            }, args.output_dir + '/' + snapshot_name + "_last")
            if epoch % args.test_every == 0:
                preds_dir = os.path.join(args.predictions, snapshot_name)
                dice_best, apls_best = evaluate_val(args, val_data_loader, apls_best, dice_best, model,
                                                    snapshot_name=snapshot_name,
                                                    current_epoch=current_epoch,
                                                    optimizer=optimizer, summary_writer=summary_writer,
                                                    predictions_dir=preds_dir, visualizer_path=args.visualizer_path,
                                                    data_dir=args.data_dir, truth_csv=args.truth_csv)
        current_epoch += 1


def evaluate_val(args, data_val, apls_best, dice_best, model, snapshot_name, current_epoch, optimizer, summary_writer,
                 predictions_dir, data_dir, truth_csv, visualizer_path):
    print("Test phase")
    model = model.eval()
    dice, apls = validate(model, data_loader=data_val, predictions_dir=predictions_dir, visualizer_path=visualizer_path,
                          data_dir=data_dir, truth_csv=truth_csv)
    if args.local_rank == 0:
        summary_writer.add_scalar('val/dice', float(dice), global_step=current_epoch)
        summary_writer.add_scalar('val/apls', float(apls), global_step=current_epoch)
        if dice > dice_best:
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'dice_best': dice,
                    'apls_best': apls,

                }, args.output_dir + snapshot_name + "_best_dice")
            dice_best = dice
        if apls > apls_best:
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'dice_best': dice,
                    'apls_best': apls,
                }, args.output_dir + snapshot_name + "_best_apls")
            apls_best = apls
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'dice_best': dice_best,
            'apls_best': apls_best,
        }, args.output_dir + snapshot_name + "_last")
        print("dice: {}, dice_best: {}".format(dice, dice_best))
        print("APLS: {}, APLS_BEST: {}".format(apls, apls_best))
    return dice_best, apls_best


def calculate_visualizer(visualizer_path, truth_csv, pred_path, img_dir):
    truth_file = truth_csv
    poly_file = pred_path

    cmd = [
        'java',
        '-jar',
        visualizer_path,
        '-truth',
        truth_file,
        '-solution',
        poly_file,
        '-no-gui',
        '-data-dir',
        img_dir
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_data, stderr_data = proc.communicate()
    lines = [line for line in stdout_data.decode('utf8').strip().split('\n')]
    overall = 0
    for i, line in enumerate(lines):
        if "Overall score :" in line:
            if "[" in line:
                overall = float(line.split("[")[-1].strip("]"))
            for l in lines[i - 14:]:
                print(l)

    return overall


def validate(net, data_loader, predictions_dir, visualizer_path, data_dir, truth_csv="oof_gt.txt"):
    os.makedirs(predictions_dir, exist_ok=True)
    dices = []
    with torch.no_grad():
        for sample in tqdm(data_loader):
            imgs = sample["image"].cuda().float()
            mask = sample["mask"].cuda().float()

            output = net(imgs)
            pred = torch.sigmoid(output)
            d = dice_round(pred[:, 10:11, ...], mask[:, 10:11, ...], t=0.5).cpu().numpy()
            for i in range(d.shape[0]):
                dices.append(d[i])
                cv2.imwrite(os.path.join(predictions_dir, sample["img_name"][i][:-4].replace("MS", "RGB") + ".png"),
                           pred[i, 10].cpu().numpy() * 255)
    pred_csv = os.path.join(Path(predictions_dir).parent, os.path.basename(predictions_dir) + ".txt")
    vectorize_dir(predictions_dir, pred_csv)
    apls = calculate_visualizer(visualizer_path, truth_csv=truth_csv, pred_path=pred_csv, img_dir=data_dir)

    return np.mean(dices), apls


def train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                local_rank):
    num_classes = conf['num_classes']
    losses = AverageMeter()
    speed_losses = AverageMeter()
    junction_losses = AverageMeter()
    dices = AverageMeter()
    iterator = tqdm(train_data_loader)
    model.train()
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in enumerate(iterator):
        imgs = sample["image"].cuda()
        masks = sample["mask"].cuda()
        out_mask = model(imgs)
        mask_band = 10
        jn_band = 11
        with torch.no_grad():
            pred = torch.sigmoid(out_mask[:, mask_band:jn_band, ...])
            d = dice_round(pred, masks[:, mask_band:jn_band, ...].contiguous(), t=0.5).item()
        dices.update(d, imgs.size(0))

        mask_loss = loss_functions["mask_loss"](out_mask[:, mask_band:jn_band, ...].contiguous(), masks[:, mask_band:jn_band, ...].contiguous())
        speed_loss = loss_functions["speed_loss"](out_mask[:, :mask_band, ...].contiguous(), masks[:, :mask_band, ...].contiguous())
        loss = speed_loss + mask_loss
        if num_classes > 8:
            junction_loss = loss_functions["junction_loss"](out_mask[:, jn_band:jn_band+1, ...].contiguous(), masks[:, jn_band:jn_band+1, ...].contiguous())
            junction_losses.update(junction_loss.item(), imgs.size(0))
            loss += junction_loss
        losses.update(loss.item(), imgs.size(0))
        speed_losses.update(speed_loss.item(), imgs.size(0))
        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss ({loss.avg:.4f}); Dice ({dice.avg:.4f}); Speed ({speed.avg:.4f}); Junction ({junction.avg:.4f}); ".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices, speed=speed_losses, junction=junction_losses))
        optimizer.zero_grad()
        if conf['fp16']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * len(train_data_loader))

    if local_rank == 0:
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)


if __name__ == '__main__':
    main()
