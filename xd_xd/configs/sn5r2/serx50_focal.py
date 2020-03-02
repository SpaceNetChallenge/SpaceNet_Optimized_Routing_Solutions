CONFIG = {
    "modelname": "serx50_focal",

    # Filepath
    "train_data_refined_dir_ims": "/wdata/input/train/images_8bit_base/PS-RGB",
    "train_data_refined_dir_masks": "/wdata/input/train/masks_base/train_mask_binned_mc",
    "test_data_refined_dir_ims": "/wdata/input/test_public/images_8bit_base/PS-RGB",
    "folds_save_path": "/wdata/input/train/folds_a.csv",

    # Used base train.py
    "model_fqn": "aa.pytorch.models.senet.SeResnext50_32x4d_upsample",
    "pretrained_model": "/pretrained/se_resnext50_32x4d-a260b3a4.pth",
    "optimizer_fqn": "torch.optim.Adam",
    "num_workers": 2,
    "num_channels": 3,
    "num_classes": 8,
    "loss": {"soft_dice": 0.25, "focal_cannab": 0.75},

    "nb_epoch": 37,
    "no_eval_period": 27,

    # mergefolds
    "num_folds": 4,

    # skeletonize
    "skeleton_band": 7,
    "skeleton_thresh": 0.3,

    "default_val_perc": 0.20,
    "min_subgraph_length_pix": 20,
    "min_spur_length_m": 1,

    "mask_width_m": "2",

    "padding": 22,
    "eval_rows": 1344,
    "eval_cols": 1344,
    "batch_size": 8,
    "iter_size": 1,
    "lr": 0.0001,
    "lr_steps": [20, 25],
    "lr_gamma": 0.2,
    "test_pad": 64,
    "epoch_size": 8,
    "predict_batch_size": 6,
    "target_cols": 512,
    "target_rows": 512,
    "warmup": 0,
    "ignore_target_size": False
}
