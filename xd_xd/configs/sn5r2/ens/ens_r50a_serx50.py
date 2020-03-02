# qsub train: 30 hours?
# qsub evaltest: 10min
# qsub mergefolds: 5min
# qsub makegraph: 10min
CONFIG = {
    "modelname": "ens_r50a_serx50",
    "ensemble_folds": [
        {
            'name': 'r50a',
            'nfolds': 4,
        },
        {
            'name': 'serx50_focal',
            'nfolds': 4,
        },
    ],

    # Filepath
    "train_data_refined_dir_ims": "/wdata/input/train/images_8bit_base/PS-RGB",
    "train_data_refined_dir_masks": "/wdata/input/train/masks_base/train_mask_binned_mc",
    "test_data_refined_dir_ims": "/wdata/input/test_public/images_8bit_base/PS-RGB",

    "num_workers": 2,
    "num_channels": 3,
    "num_classes": 8,
    "loss": {"soft_dice": 0.25, "focal_cannab": 0.75},

    # Finished after 37 epochs.
    "early_stopper_patience": 8,
    "nb_epoch": 40,

    # mergefolds
    "num_folds": 4,

    # skeletonize
    "skeleton_band": 7,
    "skeleton_thresh": 0.3,

    "default_val_perc": 0.20,
    "min_subgraph_length_pix": 20,  # 200?
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
