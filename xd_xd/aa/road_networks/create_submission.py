from pathlib import Path

import json
import os
import pandas as pd
import networkx as nx


def pkl_dir_to_wkt(pkl_dir, output_csv_path='',
                   weight_keys=['length', 'travel_time_s'],
                   verbose=False):
    """
    Create submission wkt from directory full of graph pickles
    """
    wkt_list = []

    pkl_list = sorted([z for z in os.listdir(pkl_dir) if z.endswith('.gpickle')])
    for i, pkl_name in enumerate(pkl_list):
        G = nx.read_gpickle(os.path.join(pkl_dir, pkl_name))

        # ensure an undirected graph
        print(i, "/", len(pkl_list), "num G.nodes:", len(G.nodes()))

        name_root = pkl_name.replace('PS-RGB_', '').replace('PS-MS_', '').split('.')[0]
        # AOI_root = 'AOI' + pkl_name.split('AOI')[-1]
        # name_root = AOI_root.split('.')[0].replace('PS-RGB_', '')
        print("name_root:", name_root)

        # if empty, still add to submission
        if len(G.nodes()) == 0:
            wkt_item_root = [name_root, 'LINESTRING EMPTY']
            if len(weight_keys) > 0:
                weights = [0 for w in weight_keys]
                wkt_list.append(wkt_item_root + weights)
            else:
                wkt_list.append(wkt_item_root)

        # extract geometry pix wkt, save to list
        seen_edges = set([])
        for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):
            # make sure we haven't already seen this edge
            if (u, v) in seen_edges or (v, u) in seen_edges:
                print(u, v, "already catalogued!")
                continue
            else:
                seen_edges.add((u, v))
                seen_edges.add((v, u))
            geom_pix_wkt = attr_dict['geometry_pix'].wkt

            # check edge lnegth
            if attr_dict['length'] > 5000:
                print("Edge too long!, u,v,data:", u,v,attr_dict)
                return

            if verbose:
                print(i, "/", len(G.edges()), "u, v:", u, v)
                print("  attr_dict:", attr_dict)
                print("  geom_pix_wkt:", geom_pix_wkt)

            wkt_item_root = [name_root, geom_pix_wkt]
            if len(weight_keys) > 0:
                weights = [attr_dict[w] for w in weight_keys]
                if verbose:
                    print("  weights:", weights)
                wkt_list.append(wkt_item_root + weights)
            else:
                wkt_list.append(wkt_item_root)

    if verbose:
        print("wkt_list:", wkt_list)

    # create dataframe
    if len(weight_keys) > 0:
        cols = ['ImageId', 'WKT_Pix'] + weight_keys
    else:
        cols = ['ImageId', 'WKT_Pix']

    # use 'length_m' and 'travel_time_s' instead?
    cols_new = []
    for z in cols:
        if z == 'length':
            cols_new.append('length_m')
        elif z == 'travel_time':
            cols_new.append('travel_time_s')
        else:
            cols_new.append(z)
    cols = cols_new
    # cols = [z.replace('length', 'length_m') for z in cols]
    # cols = [z.replace('travel_time', 'travel_time_s') for z in cols]
    print("cols:", cols)

    df = pd.DataFrame(wkt_list, columns=cols)
    df.loc[:, 'ImageId'] = df.ImageId.apply(lambda name: name[name.find('AOI_'):])

    print("df:", df)
    # save
    if len(output_csv_path) > 0:
        df.to_csv(output_csv_path, index=False)

    return df


def make_sub(conf):
    weight_keys = ['length', 'travel_time_s']
    verbose = False

    # output pkl
    pkl_dir = "{}/working/sp5r2/models/graphs_speed/{}".format(
        "/wdata", conf.modelname)

    # solution
    root_dir = "{}/working/sp5r2/models/solution/{}".format(
        "/wdata", conf.modelname)
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    output_csv_path = os.path.join(root_dir, 'solution.csv')

    df = pkl_dir_to_wkt(pkl_dir,
                        output_csv_path=output_csv_path,
                        weight_keys=weight_keys,
                        verbose=verbose)


def make_sub_debug(conf):
    weight_keys = ['length', 'travel_time_s', 'inferred_speed_mph']
    verbose = False

    # output pkl
    pkl_dir = "{}/working/sp5r2/models/graphs_speed/{}".format(
        "/wdata", conf.modelname)

    # solution
    root_dir = "{}/working/sp5r2/models/solution/{}".format(
        "/wdata", conf.modelname)
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    output_csv_path = os.path.join(root_dir, 'solution_debug.csv')

    df = pkl_dir_to_wkt(pkl_dir,
                        output_csv_path=output_csv_path,
                        weight_keys=weight_keys,
                        verbose=verbose)
