from collections import namedtuple

Config = namedtuple("Config", [
    "path_src",
    "path_data_root",
    "path_results_root",
    "num_channels",
    "skeleton_thresh",
    "min_subgraph_length_pix",
    "min_spur_length_m",
    "skeleton_band",
    "test_data_refined_dir",
    "test_results_dir",


    "folds_save_dir",
    "wkt_submission",
    "skeleton_dir",
    "skeleton_pkl_dir",
    "graph_dir",

    "num_classes",
])