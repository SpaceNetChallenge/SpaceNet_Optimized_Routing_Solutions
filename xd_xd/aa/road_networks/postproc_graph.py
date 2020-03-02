from collections import defaultdict

import numpy as np
import pandas as pd

from easydict import EasyDict as edict
import shapely.affinity
from shapely import wkt
import networkx as nx

import aa.road_networks.wkt_to_graph


def get_bigmap_chip_locations(aoi_name, fn_sub, df, aoi_data_path_mapping):
    cols = [
        'imname',
        'ix',
        'iy',
        'ImageId',
        'WKT_Pix',
        'length_m',
        'speed_mph',
    ]

    assert aoi_name in aoi_data_path_mapping.keys()

    df = df[(df.aoi_name == aoi_name) & (df['mode'] == 'test')]

    assert len(df) > 0

    ix_min, ix_max, iy_min, iy_max = df['ix'].min(), df['ix'].max(), df['iy'].min(), df['iy'].max()

    df = df[(df['ix'] >= ix_min) &
            (df['ix'] <= ix_max) &
            (df['iy'] >= iy_min) &
            (df['iy'] <= iy_max)]

    df_sub = pd.read_csv(fn_sub).rename(columns={'inferred_speed_mph': 'speed_mph'})
    df_sub = df_sub[df_sub.ImageId.str.startswith(aoi_name)].drop_duplicates()

    assert len(df_sub) > 0

    df_sub.loc[:, 'imname'] = df_sub.ImageId.apply(lambda x: x.split('_')[-1])
    df = df.merge(df_sub, how='left', on='imname')

    df = df.dropna(subset=['ImageId'])
    return edict(ix_min=ix_min, ix_max=ix_max, iy_min=iy_min, iy_max=iy_max), df


def construct_graph_and_node_list(df):
    wkt_list = []
    metadata_list = []

    for _, r in df.iterrows():
        wkt = shapely.affinity.translate(
            shapely.wkt.loads(r['WKT_Pix']), xoff=r['ix'] * 1300, yoff=r['iy'] * 1300).wkt
        if wkt == 'LINESTRING EMPTY':
            continue
        if wkt not in wkt_list:
            wkt_list.append(wkt)
            metadata_list.append(dict(WKT_Pix=r['WKT_Pix'],
                                      ImageId=r['ImageId'],
                                      imname=r['imname'],
                                      length_m=r['length_m'],
                                      speed_mph=r['speed_mph'],
                                      travel_time_s=r['travel_time_s']))

    node_loc_dic, edge_dic = aa.road_networks.wkt_to_graph.wkt_list_to_nodes_edges(wkt_list)  # xs, ys = shape.coords.xy
    G0 = aa.road_networks.wkt_to_graph.nodes_edges_to_G(node_loc_dic, edge_dic)

    for n, attr_dict in G0.nodes(data=True):
        x, y = attr_dict['x_pix'], attr_dict['y_pix']
        attr_dict['x'] = x
        attr_dict['y'] = y * -1  # 9nvert top-bottom

    return metadata_list, node_loc_dic, edge_dic, G0


def make_chip_graph_connected(G0,
                              node_loc_dic,
                              ix_boundary_data,
                              chip_connection_margin=20,
                              distance_margin=40):
    df_merge = pd.DataFrame([
        dict(node_id=node_id, x=x, y=y, x_in_chip=x%1300, y_in_chip=y%1300)
        for node_id, (x, y) in node_loc_dic.items()
        if ((x % 1300 < chip_connection_margin) or
            (y % 1300 < chip_connection_margin) or
            (x % 1300 >= (1300 - chip_connection_margin)) or
            (y % 1300 >= (1300 - chip_connection_margin))) and (len(G0[node_id]) > 0)])
    df_merge.loc[:, 'ix'] = np.round((df_merge['x'] - df_merge['x'] % 1300) / 1300).astype(np.int32)
    df_merge.loc[:, 'iy'] = np.round((df_merge['y'] - df_merge['y'] % 1300) / 1300).astype(np.int32)

    G1 = G0.copy()
    ix_max = ix_boundary_data.ix_max
    ix_min = ix_boundary_data.ix_min
    iy_max = ix_boundary_data.iy_max
    iy_min = ix_boundary_data.iy_min

    for index_x in range(ix_max - ix_min + 1):
        stich_base_chip_ix = ix_min + index_x
        for index_y in range(iy_max - iy_min + 1):
            stich_base_chip_iy = iy_min + index_y

            if stich_base_chip_ix < ix_max:
                df_lhs = df_merge[(df_merge['iy'] == stich_base_chip_iy) & (df_merge['ix'] == stich_base_chip_ix)]
                df_rhs = df_merge[(df_merge['iy'] == stich_base_chip_iy) & (df_merge['ix'] == stich_base_chip_ix + 1)]
                for _, r in df_lhs[df_lhs.x_in_chip >= (1300 - chip_connection_margin)].iterrows():
                    for _, r2 in df_rhs[df_rhs.x_in_chip < chip_connection_margin].iterrows():
                        dist = abs(r.y_in_chip - r2.y_in_chip)
                        if dist < distance_margin:
                            G1.add_edge(r.node_id, r2.node_id, length_pix=0)

            if stich_base_chip_iy < iy_max:
                df_lhs = df_merge[(df_merge['iy'] == stich_base_chip_iy) & (df_merge['ix'] == stich_base_chip_ix)]
                df_rhs = df_merge[(df_merge['iy'] == stich_base_chip_iy + 1) & (df_merge['ix'] == stich_base_chip_ix)]
                for _, r in df_lhs[df_lhs.y_in_chip >= (1300 - chip_connection_margin)].iterrows():
                    for _, r2 in df_rhs[df_rhs.y_in_chip < chip_connection_margin].iterrows():
                        dist = abs(r.x_in_chip - r2.x_in_chip)
                        if dist < distance_margin:
                            G1.add_edge(r.node_id, r2.node_id, length_pix=0)

    return G1


def get_connected_component_dataframe(G1):
    connected_component = []
    for i, cc in enumerate(nx.connected_component_subgraphs(G1)):
        edge_length_pix_ttl = 0
        for src, dst in cc.edges():
            edge_length_pix = sum([elem['length_pix'] for elem in cc[src][dst].values()])
            edge_length_pix_ttl += edge_length_pix
        connected_component.append({'idx': i, 'edge_length_pix_total': edge_length_pix_ttl, 'cc': cc})
    return pd.DataFrame(connected_component)


def select_registered_nodes(df_cc, lowest_length_pix_connected_component, node_loc_dic):
    small_cc_nodes = set()
    for _, r in df_cc[df_cc.edge_length_pix_total > lowest_length_pix_connected_component].iterrows():
        for v in r.cc.nodes():
            small_cc_nodes.add(v)

    area_xmax, area_ymax = max([x for x, y in node_loc_dic.values()]), max([y for x, y in node_loc_dic.values()])
    for _, r in df_cc.iterrows():
        boundary_area_flag = False
        for v in r.cc.nodes():
            x, y = node_loc_dic[v]
            if x < 650 or y < 650:
                boundary_area_flag = True
                break
            if x > area_xmax - 650 or y > area_ymax - 650:
                boundary_area_flag = True
                break
        if boundary_area_flag:
            for v in r.cc.nodes():
                small_cc_nodes.add(v)

    return small_cc_nodes


def make_refined_graph(fn_sub, edge_dic, selected_nodes, metadata_list, aoi_name):
    rows = []
    for e in edge_dic:
        edge = edge_dic[e]
        if edge['start'] not in selected_nodes:
            continue
        if edge['end'] not in selected_nodes:
            continue

        idx = edge['osmid']
        rows.append(dict(
            ImageId=metadata_list[idx]['ImageId'],
            WTK_Pix=metadata_list[idx]['WKT_Pix'],
            length_m=metadata_list[idx]['length_m'],
            travel_time_s=metadata_list[idx]['travel_time_s'],
            speed_mph=metadata_list[idx]['speed_mph'],
        ))
    df_rows_new = pd.DataFrame(rows).drop_duplicates()

    df_sub = pd.read_csv(fn_sub).rename(columns={'inferred_speed_mph': 'speed_mph'})
    aoi_name_unique_image_ids = df_sub[df_sub.ImageId.str.startswith(aoi_name)].ImageId.unique()

    rows = []
    empty_image_ids = [image_id for image_id in aoi_name_unique_image_ids if image_id not in df_rows_new.ImageId.unique()]
    for image_id in empty_image_ids:
        rows.append(dict(
            ImageId=image_id,
            WTK_Pix='LINESTRING EMPTY',
            length_m=0.0,
            travel_time_s=0.0,
            speed_mph=0.0,
        ))
    df_rows_new2 = pd.DataFrame(rows)

    df_sub_new = pd.concat([df_rows_new, df_rows_new2], sort=False).sort_values(by='ImageId')
    return df_sub_new


def main(fn_sub, fn_out, fn_out_debug, df_chiplocations, aoi_data_path_mapping):
    list_df_sub = []
    for aoi_name in sorted(aoi_data_path_mapping.keys()):
        lowest_length_pix_connected_component = 1000

        ix_boundary_data, df = get_bigmap_chip_locations(aoi_name, fn_sub, df_chiplocations, aoi_data_path_mapping)
        metadata_list, node_loc_dic, edge_dic, G0 = construct_graph_and_node_list(df)

        G1 = make_chip_graph_connected(G0,
                                       node_loc_dic,
                                       ix_boundary_data,
                                       chip_connection_margin=20,
                                       distance_margin=40)
        df_cc = get_connected_component_dataframe(G1)
        selected_nodes = select_registered_nodes(df_cc, lowest_length_pix_connected_component, node_loc_dic)
        df_sub_refined = make_refined_graph(fn_sub, edge_dic, selected_nodes, metadata_list, aoi_name)
        G3 = G1

        selected_remove_nodes = set()
        for idx, r in df_cc[df_cc.edge_length_pix_total > 1000].sort_values(by='edge_length_pix_total', ascending=False).iterrows():
            is_simple_line = True
            degree_list = []
            for node_id in r['cc']:
                degree_list.append(len(G3[node_id]))

            if np.mean(degree_list) > 2.001:
                is_simple_line = False

            if is_simple_line:
                cc_degree_set = defaultdict(set)
                for src in r['cc']:
                    for dst in G3[src]:
                        for e in G3[src][int(dst)].values():
                            if 'osmid' in e:
                                pos = list(node_loc_dic[int(src)])
                                pos = list(node_loc_dic[int(dst)])
                                for node_loc in metadata_list[e['osmid']]['WKT_Pix'][12:-1].split(', '):
                                    node_loc = node_loc.split(' ')
                                    node_loc = (int(node_loc[0]), int(node_loc[1]))
                                    cc_degree_set[metadata_list[e['osmid']]['ImageId']].add(node_loc)

                neighbor_node_list = []

                for imageid in cc_degree_set.keys():
                    for _, r2 in df[df['ImageId'] == imageid].iterrows():
                        for node_loc in r2['WKT_Pix'][12:-1].split(', '):
                            node_loc = node_loc.split(' ')
                            node_loc = (int(node_loc[0]), int(node_loc[1]))
                            rhs_node_loc_set = cc_degree_set[imageid]
                            if node_loc not in rhs_node_loc_set:
                                for rhs_node_loc in rhs_node_loc_set:
                                    dist = np.sqrt((rhs_node_loc[0] - node_loc[0]) ** 2 +
                                                   (rhs_node_loc[1] - node_loc[1]) ** 2)
                                    neighbor_node_list.append((dist, node_loc))

                if len(neighbor_node_list) > 0:
                    min_dist, _ = min(neighbor_node_list)
                    if min_dist > 200:
                        for node_id in list(r['cc']):
                            selected_remove_nodes.add(node_id)

                        print(min_dist, r['idx'], cc_degree_set.keys())
                        print([len(G3[node_id]) for node_id in r['cc']])

        selected_nodes = [n for n in selected_nodes if n not in selected_remove_nodes]
        df_sub_refined = make_refined_graph(fn_sub, edge_dic, selected_nodes, metadata_list, aoi_name)

        list_df_sub.append(df_sub_refined)

    df_sub = pd.concat(list_df_sub, sort=False)
    df_sub[[
        'ImageId',
        'WTK_Pix',
        'length_m',
        'travel_time_s',
        'speed_mph',
    ]].rename(columns={'WTK_Pix': 'WKT_Pix'}).to_csv(fn_out_debug, index=False)

    df_sub[[
        'ImageId',
        'WTK_Pix',
        'length_m',
        'travel_time_s',
    ]].rename(columns={'WTK_Pix': 'WKT_Pix'}).to_csv(fn_out, index=False)


def main_stage2(fn_sub_stg2, fn_out_stg2, fn_out_debug_stg2, df_chiplocations, aoi_data_path_mapping):
    list_df_sub = []
    for aoi_name in sorted(aoi_data_path_mapping.keys()):
        lowest_length_pix_connected_component = 10000
        ix_boundary_data, df = get_bigmap_chip_locations(aoi_name,
                                                         fn_sub_stg2,
                                                         df_chiplocations,
                                                         aoi_data_path_mapping)
        metadata_list, node_loc_dic, edge_dic, G0 = construct_graph_and_node_list(df)
        G1 = make_chip_graph_connected(G0,
                                       node_loc_dic,
                                       ix_boundary_data,
                                       chip_connection_margin=20,
                                       distance_margin=40)
        df_cc = get_connected_component_dataframe(G1)
        selected_nodes = select_registered_nodes(df_cc, lowest_length_pix_connected_component, node_loc_dic)
        df_sub_refined = make_refined_graph(fn_sub_stg2,
                                            edge_dic,
                                            selected_nodes,
                                            metadata_list,
                                            aoi_name)

        list_df_sub.append(df_sub_refined)

    df_sub = pd.concat(list_df_sub, sort=False)
    df_sub[[
        'ImageId',
        'WTK_Pix',
        'length_m',
        'travel_time_s',
        'speed_mph',
    ]].rename(columns={'WTK_Pix': 'WKT_Pix'}).to_csv(fn_out_debug_stg2, index=False)

    df_sub[[
        'ImageId',
        'WTK_Pix',
        'length_m',
        'travel_time_s',
    ]].rename(columns={'WTK_Pix': 'WKT_Pix'}).to_csv(fn_out_stg2, index=False)
