import os
# os.environ["MKL_NUM_THREADS"] = "2" 
# os.environ["NUMEXPR_NUM_THREADS"] = "2" 
# os.environ["OMP_NUM_THREADS"] = "2" 

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

from tqdm import tqdm
import timeit
import cv2

from zoo.models import Dpn92_9ch_Unet, SeResNext50_Unet_9ch, Res34_9ch_Unet

from utils import *

from sklearn.model_selection import train_test_split

import ntpath

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

test_png = '/wdata/test_png'
test_png2 = '/wdata/test_png_5_3_0'
test_png3 = '/wdata/test_png_pan_6_7'

models_folder = '/wdata/weights'
output_folder = '/wdata/test_pred'

speed_bins = [15, 18.75, 20, 25, 30, 35, 45, 55, 65]


class TestData(Dataset):
    def __init__(self, files):
        super().__init__()
        self.files = files

    def __len__(self):
        return len(self.files) * 4

    def __getitem__(self, idx):
        fn = self.files[idx // 4]

        img_id = ntpath.basename(fn)[0:-4]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img2 = cv2.imread(fn.replace('test_png', 'test_png_5_3_0'), cv2.IMREAD_COLOR)
        img3 = cv2.imread(fn.replace('test_png', 'test_png_pan_6_7'), cv2.IMREAD_COLOR)

        img = np.pad(img, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        img2 = np.pad(img2, ((6, 6), (6, 6), (0, 0)), mode='reflect')
        img3 = np.pad(img3, ((6, 6), (6, 6), (0, 0)), mode='reflect')

        img = np.concatenate([img, img2, img3], axis=2)

        if idx % 4 == 1:
            img = img[::-1, ...]
        elif idx % 4 == 2:
            img = img[:, ::-1, ...]
        elif idx % 4 == 3:
            img = img[::-1, ::-1, ...]

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        sample = {'img': img, 'img_id': img_id}
        return sample



if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(output_folder, exist_ok=True)

    # vis_dev = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    cudnn.benchmark = True
        
    test_batch_size = 8 # 12

    all_files = []
    for f in listdir(test_png):
        if '.png' in f:
            all_files.append(path.join(test_png, f))

    test_data = TestData(all_files)

    test_data_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=16, shuffle=False)

    models = []

    for fold in [3, 4]:
        model = nn.DataParallel(Dpn92_9ch_Unet(pretrained=False)).cuda()

        snap_to_load = 'dpn92_9ch_{}_1_best'.format(fold)
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (best_score {})"
                .format(snap_to_load, checkpoint['best_score']))
                        
        model = model.eval()
        models.append(model)

    for fold in [5, 6]:
        model = nn.DataParallel(SeResNext50_Unet_9ch(pretrained=None)).cuda()

        snap_to_load = 'res50_9ch_{}_0_best'.format(fold)
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (best_score {})"
                .format(snap_to_load, checkpoint['best_score']))
                        
        model = model.eval()
        models.append(model)

    for fold in [7, 8]:
        model = nn.DataParallel(Res34_9ch_Unet(pretrained=False)).cuda()

        snap_to_load = 'res34_9ch_full_{}_0_best'.format(fold)
        checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict:
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (best_score {})"
                .format(snap_to_load, checkpoint['best_score']))
                        
        model = model.eval()
        models.append(model)


    with torch.no_grad():
        for sample in tqdm(test_data_loader):
            imgs = sample["img"].cuda(non_blocking=True)
            img_ids =  sample["img_id"]
            
            msk_preds = []
            speed_cont_preds = []
            msk_speed_preds = []
            ids = []
            for i in range(0, len(img_ids), 4):
                img_id = img_ids[i]
                ids.append(img_id)
                msk_preds.append([])
                speed_cont_preds.append([])
                msk_speed_preds.append([])

            for model  in models:
                out = model(imgs)
                msk_pred0 = torch.sigmoid(out[:, :2, ...]).cpu().numpy()
                speed_cont_pred0 = out[:, 2, ...].cpu().numpy()
                speed_cont_pred0[speed_cont_pred0 < 0] = 0
                speed_cont_pred0[speed_cont_pred0 > 1] = 1
                msk_speed_pred0 = torch.softmax(out[:, 3:, ...], dim=1).cpu().numpy()

                for i in range(len(ids)):
                    msk_preds[i].append(msk_pred0[i*4 + 0])
                    msk_preds[i].append(msk_pred0[i*4 + 1, :, ::-1, :])
                    msk_preds[i].append(msk_pred0[i*4 + 2, :, :, ::-1])
                    msk_preds[i].append(msk_pred0[i*4 + 3, :, ::-1, ::-1])
                    
                    speed_cont_preds[i].append(speed_cont_pred0[i*4 + 0])
                    speed_cont_preds[i].append(speed_cont_pred0[i*4 + 1, ::-1, :])
                    speed_cont_preds[i].append(speed_cont_pred0[i*4 + 2, :, ::-1])
                    speed_cont_preds[i].append(speed_cont_pred0[i*4 + 3, ::-1, ::-1])

                    msk_speed_preds[i].append(msk_speed_pred0[i*4 + 0])
                    msk_speed_preds[i].append(msk_speed_pred0[i*4 + 1, :, ::-1, :])
                    msk_speed_preds[i].append(msk_speed_pred0[i*4 + 2, :, :, ::-1])
                    msk_speed_preds[i].append(msk_speed_pred0[i*4 + 3, :, ::-1, ::-1])


            for i in range(len(ids)):
                msk_pred = np.asarray(msk_preds[i])
                msk_pred = msk_pred.mean(axis=0)
                speed_cont_pred = np.asarray(speed_cont_preds[i])
                speed_cont_pred = speed_cont_pred.mean(axis=0)
                msk_speed_pred = np.asarray(msk_speed_preds[i])
                msk_speed_pred = msk_speed_pred.mean(axis=0)

                pred_img0 = np.concatenate([msk_pred[0, ..., np.newaxis], msk_pred[1, ..., np.newaxis], np.zeros_like(msk_pred[0, ..., np.newaxis])], axis=2)
                cv2.imwrite(path.join(output_folder,  ids[i] + '.png'), (pred_img0 * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(path.join(output_folder,  ids[i] + '_speed0.png'), (msk_speed_pred[:3].transpose(1, 2, 0) * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(path.join(output_folder,  ids[i] + '_speed1.png'), (msk_speed_pred[3:6].transpose(1, 2, 0) * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(path.join(output_folder,  ids[i] + '_speed2.png'), (msk_speed_pred[6:9].transpose(1, 2, 0) * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(path.join(output_folder,  ids[i] + '_speed_cont.png'), (speed_cont_pred * 255).astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60)) 
