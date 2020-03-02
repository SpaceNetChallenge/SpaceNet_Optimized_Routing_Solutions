import os
import sys
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.nn.functional as F
# torch.backends.cudnn.benchmark = True
import tqdm
from torch.serialization import SourceChangeWarning
import warnings
# import torchsummary
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader

from aa.cresi.net.dataset.neural_dataset import SequentialDataset


class flip(object):
    FLIP_NONE=0
    FLIP_LR=1
    FLIP_FULL=2


def flip_tensor_lr(batch):
    columns = batch.data.size()[-1]
    index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(columns)))).cuda())
    return batch.index_select(3, index)


def flip_tensor_ud(batch):
    rows = batch.data.size()[-2]
    index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(rows)))).cuda())
    return batch.index_select(2, index)


def to_numpy(batch):
    return np.moveaxis(batch.data.cpu().numpy(), 1, -1)


def predict(model, batch, flips=flip.FLIP_NONE, verbose=False):
    with torch.no_grad():
        pred1 = torch.sigmoid(model(batch))

    if verbose:
        print("  eval.py - predict() - batch.shape:", batch.shape)
        print("  eval.py - predict() - pred1.shape:", pred1.shape)

    if flips > flip.FLIP_NONE:
        with torch.no_grad():
            pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))
        masks = [pred1, pred2]
        if flips > flip.FLIP_LR:
            with torch.no_grad():
                pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
                pred4 = flip_tensor_ud(flip_tensor_lr(model(flip_tensor_ud(flip_tensor_lr(batch)))))
            masks.extend([pred3, pred4])
        masks = list(map(F.sigmoid, masks))
        new_mask = torch.mean(torch.stack(masks, 0), 0)
        return to_numpy(new_mask)
    return to_numpy(pred1)


def read_model(path_model_weights, fold):
    print ("Running eval.read_model()...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SourceChangeWarning)
        model = torch.load(os.path.join(path_model_weights, 'fold{}_best.pth'.format(fold)))
        model.eval()
        return model


class Evaluator(object):
    def __init__(self, config, ds, save_dir='', test=False, flips=0,
                 num_workers=0, border=12, val_transforms=None,
                 weight_dir='', save_im_gdal_format=True):
        self.config = config
        self.ds = ds
        self.test = test
        self.flips = flips
        self.num_workers = num_workers

        self.current_prediction = None
        self.need_to_save = False
        self.border = border

        self.save_dir = save_dir
        self.weight_dir = weight_dir
        self.save_im_gdal_format = save_im_gdal_format

        self.val_transforms = val_transforms
        os.makedirs(self.save_dir, exist_ok=True)

    def predict(self, fold, val_indexes, weight_dir, verbose=False):
        print ("run eval.Evaluator.predict()...")
        prefix = ('fold' + str(fold) + "_") if (self.test and fold is not None) else ""
        print ("prefix:", prefix)
        print ("Creating datasets within pytorch_utils/eval.py()...")
        val_dataset = SequentialDataset(self.ds, val_indexes, stage='test', config=self.config, transforms=self.val_transforms)
        val_dl = PytorchDataLoader(val_dataset, batch_size=self.config.predict_batch_size, num_workers=self.num_workers, drop_last=False)
        print ("len val_dl:", len(val_dl))
        model = read_model(weight_dir, fold)
        pbar = tqdm.tqdm(val_dl, total=len(val_dl))
        for data in pbar:
            # print("data['image'].shape:", data['image'].shape)
            with torch.no_grad():
                samples = torch.autograd.Variable(data['image'], volatile=True).cuda()
                predicted = predict(model, samples, flips=self.flips)
            self.process_batch(predicted, model, data, prefix=prefix)
        self.post_predict_action(prefix=prefix)

    def cut_border(self, image):
        if image is None:
            return None
        return image if not self.border else image[self.border:-self.border, self.border:-self.border, ...]

    def on_image_constructed(self, name, prediction, prefix=""):
        prediction = self.cut_border(prediction)
        prediction = np.squeeze(prediction)
        self.save(name, prediction, prefix=prefix)

    def save(self, name, prediction, prefix=""):
        raise NotImplementedError

    def process_batch(self, predicted, model, data, prefix=""):
        raise NotImplementedError

    def post_predict_action(self, prefix):
        pass
