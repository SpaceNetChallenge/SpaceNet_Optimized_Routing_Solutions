import os
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

# from apex import amp

from adamw import AdamW
from losses import dice_round, ComboLoss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import Res34_9ch_Unet

from imgaug import augmenters as iaa

from utils import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

import gc

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

train_png = '/wdata/train_png'
train_png2 = '/wdata/train_png_5_3_0'
train_png3 = '/wdata/train_png_pan_6_7'

masks_dir = '/wdata/masks'

models_folder = '/wdata/weights'
# val_output_folder = 'res34_9ch_val_0'

speed_bins = [15, 18.75, 20, 25, 30, 35, 45, 55, 65]

cities = [('AOI_7_Moscow', '/home/hdd/SN5_roads/train/AOI_7_Moscow', 'train_AOI_7_Moscow_geojson_roads_speed_wkt_weighted_simp.csv'),
          ('AOI_8_Mumbai', '/home/hdd/SN5_roads/train/AOI_8_Mumbai', 'train_AOI_8_Mumbai_geojson_roads_speed_wkt_weighted_simp.csv'),
          ('AOI_2_Vegas', '/home/hdd/SN3_roads/train/AOI_2_Vegas', 'train_AOI_2_Vegas_geojson_roads_speed_wkt_weighted_simp.csv'),
          ('AOI_3_Paris', '/home/hdd/SN3_roads/train/AOI_3_Paris', 'train_AOI_3_Paris_geojson_roads_speed_wkt_weighted_simp.csv'),
          ('AOI_4_Shanghai', '/home/hdd/SN3_roads/train/AOI_4_Shanghai', 'train_AOI_4_Shanghai_geojson_roads_speed_wkt_weighted_simp.csv'),
          ('AOI_5_Khartoum', '/home/hdd/SN3_roads/train/AOI_5_Khartoum', 'train_AOI_5_Khartoum_geojson_roads_speed_wkt_weighted_simp.csv')]

cities_idxs = {}
for i in range(len(cities)):
    cities_idxs[cities[i][0]] = i


input_shape = (704, 704) # (384, 384)

train_files = []
for f in listdir(train_png):
    if '.png' in f:
        train_files.append(f)



class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        img_id = train_files[_idx]

        img = cv2.imread(path.join(train_png, img_id), cv2.IMREAD_COLOR)
        img2 = cv2.imread(path.join(train_png2, img_id), cv2.IMREAD_COLOR)
        img3 = cv2.imread(path.join(train_png3, img_id), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(path.join(masks_dir, img_id), cv2.IMREAD_COLOR)
        msk1 = cv2.imread(path.join(masks_dir, img_id.replace('.png', '_speed0.png')), cv2.IMREAD_COLOR)
        msk2 = cv2.imread(path.join(masks_dir, img_id.replace('.png', '_speed1.png')), cv2.IMREAD_COLOR)
        msk3 = cv2.imread(path.join(masks_dir, img_id.replace('.png', '_speed2.png')), cv2.IMREAD_COLOR)
        msk4 = cv2.imread(path.join(masks_dir, img_id.replace('.png', '_speed_cont.png')), cv2.IMREAD_UNCHANGED)

        #TODO finally finetune Moscow (only?) without flips and rotations!
        

        if (('Moscow' not in img_id) and ('Mumbai' not in img_id) and (random.random() > 0.8)) or (random.random() > 0.9):
            img = img[::-1, ...]
            img2 = img2[::-1, ...]
            img3 = img3[::-1, ...]
            msk0 = msk0[::-1, ...]
            msk1 = msk1[::-1, ...]
            msk2 = msk2[::-1, ...]
            msk3 = msk3[::-1, ...]
            msk4 = msk4[::-1, ...]

        if (('Moscow' not in img_id) and ('Mumbai' not in img_id) and (random.random() > 0.8)) or (random.random() > 0.9):
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)
                img2 = np.rot90(img2, k=rot)
                img3 = np.rot90(img3, k=rot)
                msk0 = np.rot90(msk0, k=rot)
                msk1 = np.rot90(msk1, k=rot)
                msk2 = np.rot90(msk2, k=rot)
                msk3 = np.rot90(msk3, k=rot)
                msk4 = np.rot90(msk4, k=rot)
                    
        if random.random() > 0.95:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            img2 = shift_image(img2, shift_pnt)
            img3 = shift_image(img3, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            msk1 = shift_image(msk1, shift_pnt)
            msk2 = shift_image(msk2, shift_pnt)
            msk3 = shift_image(msk3, shift_pnt)
            msk4 = shift_image(msk4, shift_pnt)
            
        if random.random() > 0.95:
            rot_pnt =  (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                img2 = rotate_image(img2, angle, scale, rot_pnt)
                img3 = rotate_image(img3, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)
                msk1 = rotate_image(msk1, angle, scale, rot_pnt)
                msk2 = rotate_image(msk2, angle, scale, rot_pnt)
                msk3 = rotate_image(msk3, angle, scale, rot_pnt)
                msk4 = rotate_image(msk4, angle, scale, rot_pnt)

        crop_size = input_shape[0]
        if random.random() > 0.95:
            crop_size = random.randint(int(input_shape[0] / 1.1), int(input_shape[0] / 0.9))

        x0 = random.randint(0, img.shape[1] - crop_size)
        y0 = random.randint(0, img.shape[0] - crop_size)

        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        img2 = img2[y0:y0+crop_size, x0:x0+crop_size, :]
        img3 = img3[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size, :]
        msk1 = msk1[y0:y0+crop_size, x0:x0+crop_size, :]
        msk2 = msk2[y0:y0+crop_size, x0:x0+crop_size, :]
        msk3 = msk3[y0:y0+crop_size, x0:x0+crop_size, :]
        msk4 = msk4[y0:y0+crop_size, x0:x0+crop_size]
        

        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, input_shape, interpolation=cv2.INTER_LINEAR)
            img3 = cv2.resize(img3, input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)
            msk1 = cv2.resize(msk1, input_shape, interpolation=cv2.INTER_LINEAR)
            msk2 = cv2.resize(msk2, input_shape, interpolation=cv2.INTER_LINEAR)
            msk3 = cv2.resize(msk3, input_shape, interpolation=cv2.INTER_LINEAR)
            msk4 = cv2.resize(msk4, input_shape, interpolation=cv2.INTER_LINEAR)
            
        if random.random() > 0.97:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        if random.random() > 0.97:
            img2 = shift_channels(img2, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        if random.random() > 0.97:
            img3 = shift_channels(img3, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.97:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.97:
            if random.random() > 0.95:
                img = clahe(img)
            elif random.random() > 0.95:
                img = gauss_noise(img)
            elif random.random() > 0.95:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.97:
            if random.random() > 0.95:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.95:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.95:
                img = contrast(img, 0.9 + random.random() * 0.2)

        if random.random() > 0.98:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        msk = (msk0[..., :2] > 127) * 1
        bkg_msk = (np.ones_like(msk[..., :1]) - msk[..., :1]) * 255
        msk_speed = np.concatenate([msk1, msk2, msk3, bkg_msk], axis=2)
        msk_speed = (msk_speed > 127) * 1
        for i in range(9):
            for j in range(i + 1, 10):
                msk_speed[msk_speed[..., 9-i] > 0, 9-j] = 0
        lbl_speed = msk_speed.argmax(axis=2)

        msk_speed_cont = msk4 / 255
        msk_speed_cont = msk_speed_cont[..., np.newaxis]

        img = np.concatenate([img, img2, img3], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()
        msk_speed = torch.from_numpy(msk_speed.transpose((2, 0, 1))).long()
        msk_speed_cont = torch.from_numpy(msk_speed_cont.transpose((2, 0, 1))).float()
        lbl_speed = torch.from_numpy(lbl_speed.copy()).long()

        sample = {'img': img, 'msk': msk, 'msk_speed': msk_speed, 'lbl_speed': lbl_speed, 'msk_speed_cont': msk_speed_cont, 'img_id': img_id}
        return sample


    
class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        img_id = train_files[_idx]

        img = cv2.imread(path.join(train_png, img_id), cv2.IMREAD_COLOR)
        img2 = cv2.imread(path.join(train_png2, img_id), cv2.IMREAD_COLOR)
        img3 = cv2.imread(path.join(train_png3, img_id), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(path.join(masks_dir, img_id), cv2.IMREAD_COLOR)
        msk1 = cv2.imread(path.join(masks_dir, img_id.replace('.png', '_speed0.png')), cv2.IMREAD_COLOR)
        msk2 = cv2.imread(path.join(masks_dir, img_id.replace('.png', '_speed1.png')), cv2.IMREAD_COLOR)
        msk3 = cv2.imread(path.join(masks_dir, img_id.replace('.png', '_speed2.png')), cv2.IMREAD_COLOR)
        msk4 = cv2.imread(path.join(masks_dir, img_id.replace('.png', '_speed_cont.png')), cv2.IMREAD_UNCHANGED)
        img = np.pad(img, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        img2 = np.pad(img2, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        img3 = np.pad(img3, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        msk0 = np.pad(msk0, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        msk1 = np.pad(msk1, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        msk2 = np.pad(msk2, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        msk3 = np.pad(msk3, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        msk4 = np.pad(msk4, ((6, 6), (6, 6)), mode='reflect')

        msk = (msk0[..., :2] > 127) * 1
        bkg_msk = (np.ones_like(msk[..., :1]) - msk[..., :1]) * 255
        msk_speed = np.concatenate([msk1, msk2, msk3, bkg_msk], axis=2)
        msk_speed = (msk_speed > 127) * 1
        for i in range(9):
            for j in range(i + 1, 10):
                msk_speed[msk_speed[..., 9-i] > 0, 9-j] = 0
        lbl_speed = msk_speed.argmax(axis=2)

        msk_speed_cont = msk4 / 255
        msk_speed_cont = msk_speed_cont[..., np.newaxis]

        img = np.concatenate([img, img2, img3], axis=2)
        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()
        msk_speed = torch.from_numpy(msk_speed.transpose((2, 0, 1))).long()
        msk_speed_cont = torch.from_numpy(msk_speed_cont.transpose((2, 0, 1))).float()
        lbl_speed = torch.from_numpy(lbl_speed.copy()).long()

        sample = {'img': img, 'msk': msk, 'msk_speed': msk_speed, 'lbl_speed': lbl_speed, 'msk_speed_cont': msk_speed_cont, 'img_id': img_id}
        return sample




def validate(net, data_loader):
    dices0 = []
    dices1 = []

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            msks = sample["msk"].numpy()
            imgs = sample["img"].cuda(non_blocking=True)
            img_ids =  sample["img_id"]
            
            out = model(imgs)
            
            msk_pred = torch.sigmoid(out[:, :2, ...]).cpu().numpy()
            speed_cont_pred = out[:, 2, ...].cpu().numpy()
            speed_cont_pred[speed_cont_pred < 0] = 0
            speed_cont_pred[speed_cont_pred > 1] = 1
            msk_speed_pred = torch.softmax(out[:, 3:, ...], dim=1).cpu().numpy()

            pred = msk_pred > 0.5
            for j in range(msks.shape[0]):
                dices0.append(dice(msks[j, 0], pred[j, 0]))
                dices1.append(dice(msks[j, 1], pred[j, 1]))
                
                # pred_img0 = np.concatenate([msk_pred[j, 0, ..., np.newaxis], msk_pred[j, 1, ..., np.newaxis], np.zeros_like(msk_pred[j, 0, ..., np.newaxis])], axis=2)
                # cv2.imwrite(path.join(val_output_folder,  img_ids[j]), (pred_img0 * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])

                # cv2.imwrite(path.join(val_output_folder,  img_ids[j].replace('.png', '_speed0.png')), (msk_speed_pred[j, :3].transpose(1, 2, 0) * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                # cv2.imwrite(path.join(val_output_folder,  img_ids[j].replace('.png', '_speed1.png')), (msk_speed_pred[j, 3:6].transpose(1, 2, 0) * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                # cv2.imwrite(path.join(val_output_folder,  img_ids[j].replace('.png', '_speed2.png')), (msk_speed_pred[j, 6:9].transpose(1, 2, 0) * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                # cv2.imwrite(path.join(val_output_folder,  img_ids[j].replace('.png', '_speed_cont.png')), (speed_cont_pred[j] * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])

    d0 = np.mean(dices0)
    d1 = np.mean(dices1)

    print("Val Dice: {}, {}".format(d0, d1))
    return d0



def evaluate_val(data_val, best_score, model, snapshot_name, current_epoch):
    model = model.eval()
    d = validate(model, data_loader=data_val)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
        }, path.join(models_folder, snapshot_name + '_best'))
        best_score = d

    print("dice: {}\tdice_best: {}".format(d, best_score))
    return best_score



def train_epoch(current_epoch, seg_loss, ce_loss, mse_loss, model, optimizer, scheduler, train_data_loader):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    losses4 = AverageMeter()

    dices = AverageMeter()

    iterator = tqdm(train_data_loader)
    model.train()
    scheduler.step(current_epoch)
    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        msks = sample["msk"].cuda(non_blocking=True)
        msks_speed = sample["msk_speed"].cuda(non_blocking=True)
        lbls_speed = sample["lbl_speed"].cuda(non_blocking=True)
        msks_speed_cont = sample["msk_speed_cont"].cuda(non_blocking=True)
        

        out = model(imgs)

        loss1 = seg_loss(out[:, 0, ...], msks[:, 0, ...])
        loss2 = seg_loss(out[:, 1, ...], msks[:, 1, ...])

        loss3 = ce_loss(out[:, 3:, ...], lbls_speed)

        loss4 = mse_loss(out[:, 2:3, ...], msks_speed_cont)

        loss = 1.5 * loss1 + 0.05 * loss2 + 0.2 * loss3 + 0.1 * loss4

        for _i in range(3, 13):
            loss += 0.03 * seg_loss(out[:, _i, ...], msks_speed[:, _i-3, ...])

        with torch.no_grad():
            _probs = torch.sigmoid(out[:, 0, ...])
            dice_sc = 1 - dice_round(_probs, msks[:, 0, ...])

        losses.update(loss.item(), imgs.size(0))
        losses1.update(loss1.item(), imgs.size(0))
        losses2.update(loss2.item(), imgs.size(0))
        losses3.update(loss3.item(), imgs.size(0))
        losses4.update(loss4.item(), imgs.size(0))

        dices.update(dice_sc, imgs.size(0))

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} ({loss.avg:.4f}); Loss1 {loss1.val:.4f} ({loss1.avg:.4f}); Loss2 {loss2.val:.4f} ({loss2.avg:.4f}); Loss3 {loss3.val:.4f} ({loss3.avg:.4f}); Loss4 {loss4.val:.4f} ({loss4.avg:.4f}); Dice {dice.val:.4f} ({dice.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, loss2=losses2, loss3=losses3, loss4=losses4, dice=dices))
        
        optimizer.zero_grad()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

#         scheduler.step()

    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}; Loss1 {loss1.avg:.4f}; Loss2 {loss2.avg:.4f}; Loss3 {loss3.avg:.4f}; Loss4 {loss4.avg:.4f}; Dice {dice.avg:.4f}".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, loss1=losses1, loss2=losses2, loss3=losses3, loss4=losses4, dice=dices))




if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(models_folder, exist_ok=True)
    # makedirs(val_output_folder, exist_ok=True)
    
    seed = int(sys.argv[1])
    vis_dev = sys.argv[2]

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    cudnn.benchmark = True

    batch_size = 6
    val_batch_size = 4

    snapshot_name = 'res34_9ch_full_{}_0'.format(seed)

    train_idxs0, val_idxs = train_test_split(np.arange(len(train_files)), test_size=0.05, random_state=seed)

    np.random.seed(seed)
    random.seed(seed)

    train_idxs = []
    for i in train_idxs0:
        train_idxs.append(i)
        if (('Paris' in train_files[i]) or ('Khartoum' in train_files[i])) and random.random() > 0.15:
            train_idxs.append(i)
        if (('Paris' in train_files[i]) or ('Khartoum' in train_files[i])) and random.random() > 0.15:
            train_idxs.append(i)
        if (('Mumbai' in train_files[i]) or ('Moscow' in train_files[i])) and random.random() > 0.7:
            train_idxs.append(i)
    train_idxs = np.asarray(train_idxs)


    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainData(train_idxs)
    val_train = ValData(val_idxs)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(val_train, batch_size=val_batch_size, num_workers=8, shuffle=False, pin_memory=False)

    model = Res34_9ch_Unet() #.cuda()

    params = model.parameters()

    optimizer = AdamW(params, lr=0.0004, weight_decay=1e-4)
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[4, 10, 16, 24, 28, 32], gamma=0.5)

    model = nn.DataParallel(model).cuda()


    seg_loss = ComboLoss({'dice': 1.0, 'focal': 3.0}, per_image=True).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()
    mse_loss = nn.MSELoss().cuda()

    best_score = 0
    _cnt = -1
    for epoch in range(34):
        train_epoch(epoch, seg_loss, ce_loss, mse_loss, model, optimizer, scheduler, train_data_loader)
        if epoch % 2 == 0:
            _cnt += 1
            # torch.save({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'best_score': best_score,
            # }, path.join(models_folder, snapshot_name + '_{}'.format(_cnt % 3)))
            torch.cuda.empty_cache()
            best_score = evaluate_val(val_data_loader, best_score, model, snapshot_name, epoch)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
