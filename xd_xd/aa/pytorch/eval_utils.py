import os
import json
import warnings
from pathlib import Path
import importlib

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import gdal
import skimage.io
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from torch.serialization import SourceChangeWarning
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader

from aa.pytorch.dataset import (
    RawImageTypePad, ReadingImageProvider, ReadingImageListProvider)
from aa.pytorch.dataset_sp5r2 import SequentialDataset
from aa.pytorch.predict_utils import predict as predict_


def dynamic_load(model_class_fqn):
    module_name = '.'.join(model_class_fqn.split('.')[:-1])
    class_name = model_class_fqn.split('.')[-1]

    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls


class Evaluator(object):
    """
    base class for evaluators
    """
    def __init__(self,
                 config,
                 ds,
                 save_dir='',
                 test=False,
                 flips=0,
                 num_workers=0,
                 border=12,
                 val_transforms=None,
                 save_im_gdal_format=True):
        self.config = config
        self.ds = ds
        self.test = test
        self.flips = flips
        self.num_workers = num_workers

        self.current_prediction = None
        self.need_to_save = False
        self.border = border

        self.save_dir = save_dir
        self.save_im_gdal_format = save_im_gdal_format

        self.val_transforms = val_transforms
        os.makedirs(self.save_dir, exist_ok=True)

    def predict(self, fold, val_indexes, weight_dir, verbose=False):
        prefix = ''
        if fold is not None:
            prefix = f'fold{fold}_'

        val_dataset = SequentialDataset(self.ds,
                                        val_indexes,
                                        stage='test',
                                        config=self.config,
                                        transforms=self.val_transforms)
        val_dl = PytorchDataLoader(
            val_dataset,
            batch_size=self.config.predict_batch_size,
            num_workers=self.num_workers,
            drop_last=False)

        dynamic_load(self.config.model_fqn)
        model = read_model_multiband(weight_dir, fold)
        pbar = tqdm(val_dl, total=len(val_dl))
        with torch.no_grad():
            for data in pbar:
                samples = torch.autograd.Variable(data['image']).cuda()
                predicted = predict_(model, samples, flips=self.flips)
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


def CreateMultiBandGeoTiff(OutPath, Array):
    '''
    Array has shape:
        Channels, Y, X?
    '''
    driver=gdal.GetDriverByName('GTiff')
    DataSet = driver.Create(OutPath, Array.shape[2], Array.shape[1],
                            Array.shape[0], gdal.GDT_Byte,
                            ['COMPRESS=LZW'])
    for i, image in enumerate(Array, 1):
        DataSet.GetRasterBand(i).WriteArray( image )
    del DataSet

    return OutPath


class FullImageEvaluatorMultiBand(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, predicted, model, data, prefix="", verbose=False):
        names = data['image_name']
        if verbose:
            print("concrete_eval.py.process_batch()  predicted.shape;", predicted.shape,)
        for i in range(len(names)):
            self.on_image_constructed(names[i], predicted[i,...], prefix)

    def save(self, name, prediction, prefix="", #save_im_gdal_format=True,
             verbose=False):
        # AVE edit
        save_im_gdal_format = self.save_im_gdal_format
        if verbose:
            print ("concrete_eval.py: prediction.shape:", prediction.shape)
            print ("np.unique prediction:", np.unique(prediction))
        if len(prediction.shape) == 2:
            cv2.imwrite(os.path.join(self.save_dir, prefix + name), (prediction * 255).astype(np.uint8))
        else:

            # skimage reads in (channels, h, w) for multi-channel
            # assume less than 20 channels
            #print ("mask_channels.shape:", mask_channels.shape)
            if prediction.shape[0] > 20:
                #print ("mask_channels.shape:", mask_channels.shape)
                mask = np.moveaxis(prediction, -1, 0)
            else:
                mask = prediction
            if verbose:
                print ("concrete_eval.py: mask.shape:", mask.shape)

            # save with skimage
            outfile_sk = os.path.join(self.save_dir, prefix + name)
            if verbose:
                print ("name:", name)
                print ("mask.shape:", mask.shape)
                print ("prediction.shape:", prediction.shape)
                print ("outfile_sk:", outfile_sk)
            skimage.io.imsave(outfile_sk, (mask * 255).astype(np.uint8),
                              compress=1)

            # also save with gdal?
            if save_im_gdal_format:
                save_dir_gdal = os.path.join(self.save_dir + '_gdal')
                #print ("save_dir_gdal:", save_dir_gdal)
                os.makedirs(save_dir_gdal, exist_ok=True)
                CreateMultiBandGeoTiff(os.path.join(save_dir_gdal, prefix + name), (mask * 255).astype(np.uint8))


def read_model_multiband(path_model_weights, fold):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SourceChangeWarning)
        model = torch.load(os.path.join(path_model_weights,
                                        f'fold{fold}_best.pth'))
        model.eval()
        return model


class FullImageEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, predicted, model, data, prefix=""):
        names = data['image_name']
        for i in range(len(names)):
            self.on_image_constructed(names[i], predicted[i,...], prefix)

    def save(self, name, prediction, prefix=""):
        im_out_path = Path(self.save_dir) / (prefix + name)
        im_out_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(im_out_path),
                    (prediction * 255).astype(np.uint8))


class FullImageMultiChannelEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, predicted, model, data, prefix=""):
        names = data['image_name']
        for i in range(len(names)):
            self.on_image_constructed(names[i], predicted[i,...], prefix)

    def save(self, name, prediction, prefix=""):
        if prediction.shape[-1] == 5:
            name_ch1 = name.replace('.png', '_ch123.png')
            name_ch2 = name.replace('.png', '_ch345.png')

            im_out_path = Path(self.save_dir) / (prefix + name_ch1)
            im_out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(im_out_path),
                        (prediction[..., :3] * 255).astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

            im_out_path = Path(self.save_dir) / (prefix + name_ch2)
            cv2.imwrite(str(im_out_path),
                        (prediction[..., 2:5] * 255).astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])
        elif prediction.shape[-1] == 8:
            name_ch1 = name.replace('.png', '_ch123.png')
            name_ch2 = name.replace('.png', '_ch456.png')
            name_ch3 = name.replace('.png', '_ch678.png')

            im_out_path = Path(self.save_dir) / (prefix + name_ch1)
            im_out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(im_out_path),
                        (prediction[..., :3] * 255).astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

            im_out_path = Path(self.save_dir) / (prefix + name_ch2)
            cv2.imwrite(str(im_out_path),
                        (prediction[..., 3:6] * 255).astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

            im_out_path = Path(self.save_dir) / (prefix + name_ch3)
            cv2.imwrite(str(im_out_path),
                        (prediction[..., 5:8] * 255).astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])


def read_model(config, fold):
    # model = nn.DataParallel(torch.load(os.path.join('..', 'weights', project, 'fold{}_best.pth'.format(fold))))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SourceChangeWarning)
        model = torch.load(os.path.join(config.results_dir,
                                        'weights',
                                        config.folder,
                                        f'fold{fold}_last.pth'))
        model.eval()
        return model


def eval_roads(config,
               args_fold=0,
               is_test=True):
    paths = {
        'masks': os.path.join(config.dataset_path, 'masks', config.maskname),
        'images': os.path.join(config.dataset_path, 'images', config.imagename)
    }
    df_fold = pd.read_csv(config.fn_folds, index_col=0)

    # Validation data provider (for each cvfold)
    if is_test is False:
        assert args_fold is not None

        filename_list = [
            '{}/{}.png'.format(
                r['aoi_name'],
                r['imname'],
            )
            for _, r in df_fold[df_fold.fold == args_fold].iterrows()
        ]
        ds = ReadingImageListProvider(RawImageTypePad, paths, filename_list)

        # Prep evaluator
        n_folds = len(df_fold.fold.unique())
        folds = [([], list(range(len(ds)))) for i in range(n_folds)]
        num_workers = 0 if os.name == 'nt' else 2
        if config.num_classes > 1:
            keval = FullImageMultiChannelEvaluator(config,
                                                   ds,
                                                   test=is_test,
                                                   flips=3,
                                                   num_workers=num_workers,
                                                   border=22)
        else:
            keval = FullImageEvaluator(config,
                                       ds,
                                       test=is_test,
                                       flips=3,
                                       num_workers=num_workers,
                                       border=22)
        for fold, (t, e) in enumerate(folds):
            if args_fold is not None and int(args_fold) != fold:
                continue
            keval.predict(fold, e)
    else:
        aoi_names = [
            'AOI_7_Moscow',
            'AOI_8_Mumbai',
            'AOI_9_San_Juan',
        ]
        filename_list = []
        for aoi_name in aoi_names:
            for fp in (Path(paths['images']) / aoi_name).glob('./*.png'):
                filename_list.append('{}/{}.png'.format(aoi_name, fp.stem))

        ds = ReadingImageListProvider(RawImageTypePad, paths, filename_list)

        # Prep evaluator
        n_folds = len(df_fold.fold.unique())
        folds = [([], list(range(len(ds)))) for i in range(n_folds)]
        num_workers = 2
        if config.num_classes > 1:
            keval = FullImageMultiChannelEvaluator(config,
                                                   ds,
                                                   test=True,
                                                   flips=3,
                                                   num_workers=num_workers,
                                                   border=22)
        else:
            keval = FullImageEvaluator(config,
                                       ds,
                                       test=True,
                                       flips=3,
                                       num_workers=num_workers,
                                       border=22)
        fold, ids = args_fold, list(range(len(ds)))
        keval.predict(fold, ids)
