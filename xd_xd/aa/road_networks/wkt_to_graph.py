import os
import time
import utm

import numpy as np
import networkx as nx
import osmnx as ox
import shapely
from shapely.geometry import mapping, Point, LineString
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt


def wkt_to_graph(wkt_list, im_file, conf, out_graph_file):
    min_subgraph_length_pix = 300
    verbose = False
    super_verbose = False
    make_plots = False
    save_shapefiles = False
    pickle_protocol = 4

    if (len(wkt_list) == 0) or (wkt_list[0] == 'LINESTRING EMPTY'):
        return None

    try:
        G = wkt_to_G(wkt_list,
                     im_file=im_file,
                     min_subgraph_length_pix=min_subgraph_length_pix,
                     verbose=super_verbose)
        if len(G.nodes()) == 0:
            return None
    except Exception as e:
        print('Exception in wkt_to_G: {}, {}'.format(
            str(e), out_graph_file))
        return None

    node = list(G.nodes())[-1]
    if verbose:
        print(node, 'random node props:', G.nodes[node])

    # print an edge
    edge_tmp = list(G.edges())[-1]
    if verbose:
        print (edge_tmp, "random edge props:", G.edges([edge_tmp[0], edge_tmp[1]])) #G.edge[edge_tmp[0]][edge_tmp[1]])

    nx.write_gpickle(G, out_graph_file, protocol=pickle_protocol)

    # save shapefile as well?
    if save_shapefiles:
        ox.save_graph_shapefile(G,
                                filename=image_id.split('.')[0] ,
                                folder=graph_dir, encoding='utf-8')

    # plot, if desired
    if make_plots:
        outfile_plot = 'debug_ox.png'
        if verbose:
            print ("Plotting graph...")
            print ("outfile_plot:", outfile_plot)
        ox.plot_graph(G, fig_height=9, fig_width=9, 
                      #save=True, filename=outfile_plot, margin=0.01)
                      )
        #plt.tight_layout()
        plt.savefig(outfile_plot, dpi=400)


def wkt_to_G(wkt_list,
             im_file=None,
             min_subgraph_length_pix=30,
             simplify_graph=True,
             verbose=False):
    if verbose:
        print ("Running wkt_list_to_nodes_edges()...")
    node_loc_dic, edge_dic = wkt_list_to_nodes_edges(wkt_list)

    if verbose:
        print ("Creating G...")
    G0 = nodes_edges_to_G(node_loc_dic, edge_dic)
    if verbose:
        print ("  len(G.nodes():", len(G0.nodes()))
        print ("  len(G.edges():", len(G0.edges()))

    if verbose:
        print ("Clean out short subgraphs")
    G0 = clean_sub_graphs(G0, min_length=min_subgraph_length_pix,
                     max_nodes_to_skip=30,
                     weight='length_pix', verbose=False,
                     super_verbose=False)

    if len(G0) == 0:
        return G0

    # geo coords
    if im_file:
        if verbose:
            print ("Running get_node_geo_coords()...")
        G1 = get_node_geo_coords(G0, im_file, verbose=verbose)

        if verbose:
            print ("Running get_edge_geo_coords()...")
        G1 = get_edge_geo_coords(G1, im_file, verbose=verbose)

        if verbose:
            print ("projecting graph...")
        G_projected = ox.project_graph(G1)

        Gout = G_projected #G_simp
    else:
        Gout = G0

    if simplify_graph:
        if verbose:
            print ("Simplifying graph")
        G0 = ox.simplify_graph(Gout.to_directed())
        G0 = G0.to_undirected()
        Gout = ox.project_graph(G0)
        if verbose:
            print ("Merge 'geometry' linestrings...")
        keys_tmp = ['geometry_pix', 'geometry_latlon_wkt', 'geometry_utm_wkt']
        for key_tmp in keys_tmp:
            if verbose:
                print ("Merge", key_tmp, "...")
            for i,(u,v,attr_dict) in enumerate(Gout.edges(data=True)):
                if (i % 10000) == 0:
                    if verbose:
                        print (i, u , v)
                geom = attr_dict[key_tmp]
                #print (i, u, v, "geom:", geom)
                #print ("  type(geom):", type(geom))

                if type(geom) == list:
                    # check if the list items are wkt strings, if so, create
                    #   linestrigs
                    if (type(geom[0]) == str):# or (type(geom_pix[0]) == unicode):
                        geom = [shapely.wkt.loads(ztmp) for ztmp in geom]
                    # merge geoms
                    #geom = shapely.ops.linemerge(geom)
                    #attr_dict[key_tmp] =  geom
                    attr_dict[key_tmp] = shapely.ops.linemerge(geom)
                elif type(geom) == str:
                    attr_dict[key_tmp] = shapely.wkt.loads(geom)
                else:
                    pass

        # assign 'geometry' tag to geometry_utm_wkt
        for i,(u,v,attr_dict) in enumerate(Gout.edges(data=True)):
            if verbose:
                print ("Create 'geometry' field in edges...")
            #geom_pix = attr_dict[key_tmp]
            line = attr_dict['geometry_utm_wkt']
            if type(line) == str:# or type(line) == unicode:
                attr_dict['geometry'] = shapely.wkt.loads(line)
            else:
                attr_dict['geometry'] = attr_dict['geometry_utm_wkt']
            # update wkt_pix?
            #print ("attr_dict['geometry_pix':", attr_dict['geometry_pix'])
            attr_dict['wkt_pix'] = attr_dict['geometry_pix'].wkt

            # update 'length_pix'
            attr_dict['length_pix'] = np.sum([attr_dict['length_pix']])


        Gout = ox.project_graph(Gout)

    if verbose:
        # get a few stats (and set to graph properties)
        print("Number of nodes: {}".format(len(Gout.nodes())))
        print("Number of edges: {}".format(len(Gout.edges())))
        #print ("Number of nodes:", len(Gout.nodes()))
        #print ("Number of edges:", len(Gout.edges()))
    Gout.graph['N_nodes'] = len(Gout.nodes())
    Gout.graph['N_edges'] = len(Gout.edges())

    # get total length of edges
    tot_meters = 0
    for i,(u,v,attr_dict) in enumerate(Gout.edges(data=True)):
        tot_meters  += attr_dict['length']
    if verbose:
        print ("Length of edges (km):", tot_meters/1000)
    Gout.graph['Tot_edge_km'] = tot_meters/1000

    if verbose:
        print ("G.graph:", Gout.graph)

    return Gout


def wkt_list_to_nodes_edges(wkt_list):
    '''Convert wkt list to nodes and edges
    Make an edge between each node in linestring. Since one linestring
    may contain multiple edges, this is the safest approach'''

    node_loc_set = set()    # set of edge locations
    node_loc_dic = {}       # key = node idx, val = location
    node_loc_dic_rev = {}   # key = location, val = node idx
    edge_loc_set = set()    # set of edge locations
    edge_dic = {}           # edge properties
    node_iter = 0
    edge_iter = 0

    for i,lstring in enumerate(wkt_list):
        # get lstring properties
        shape = shapely.wkt.loads(lstring)
        xs, ys = shape.coords.xy
        length_orig = shape.length

        # iterate through coords in line to create edges between every point
        for j,(x,y) in enumerate(zip(xs, ys)):
            loc = (x,y)
            # for first item just make node, not edge
            if j == 0:
                # if not yet seen, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1

            # if not first node in edge, retrieve previous node and build edge
            else:
                prev_loc = (xs[j-1], ys[j-1])
                #print ("prev_loc:", prev_loc)
                prev_node = node_loc_dic_rev[prev_loc]

                # if new, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                # if seen before, retrieve node properties
                else:
                    node = node_loc_dic_rev[loc]

                # add edge, which is start_node to end_node
                edge_loc = (loc, prev_loc)
                edge_loc_rev = (prev_loc, loc)
                # shouldn't be duplicate edges, so break if we see one
                if (edge_loc in edge_loc_set) or (edge_loc_rev in edge_loc_set):
                    print ("Oops, edge already seen, returning:", edge_loc)
                    return

                # get distance to prev_loc and current loc
                proj_prev = shape.project(Point(prev_loc))
                proj = shape.project(Point(loc))
                # edge length is the diffence of the two projected lengths
                #   along the linestring
                edge_length = abs(proj - proj_prev)
                # make linestring
                line_out = LineString([prev_loc, loc])
                line_out_wkt = line_out.wkt

                edge_props = {'start': prev_node,
                              'start_loc_pix': prev_loc,
                              'end': node,
                              'end_loc_pix': loc,
                              'length_pix': edge_length,
                              'wkt_pix': line_out_wkt,
                              'geometry_pix': line_out,
                              'osmid': i}
                #print ("edge_props", edge_props)
                edge_loc_set.add(edge_loc)
                edge_dic[edge_iter] = edge_props
                edge_iter += 1

    return node_loc_dic, edge_dic


def nodes_edges_to_G(node_loc_dic, edge_dic, name='glurp'):
    '''Take output of wkt_list_to_nodes_edges(wkt_list) and create networkx
    graph'''

    G = nx.MultiDiGraph()
    # set graph crs and name
    G.graph = {'name': name,
               'crs': {'init': 'epsg:4326'}
               }

    # add nodes
    #for key,val in node_loc_dic.iteritems():
    for key in node_loc_dic.keys():
        val = node_loc_dic[key]
        attr_dict = {'osmid': key,
                     'x_pix': val[0],
                     'y_pix': val[1]}
        G.add_node(key, **attr_dict)

    # add edges
    #for key,val in edge_dic.iteritems():
    for key in edge_dic.keys():
        val = edge_dic[key]
        attr_dict = val
        u = attr_dict['start']
        v = attr_dict['end']
        #attr_dict['osmid'] = str(i)

        #print ("nodes_edges_to_G:", u, v, "attr_dict:", attr_dict)
        if type(attr_dict['start_loc_pix']) == list:
            return

        G.add_edge(u, v, **attr_dict)

        ## always set edge key to zero?  (for nx 1.X)
        ## THIS SEEMS NECESSARY FOR OSMNX SIMPLIFY COMMAND
        #G.add_edge(u, v, key=0, attr_dict=attr_dict)
        ##G.add_edge(u, v, key=key, attr_dict=attr_dict)

    #G1 = ox.simplify_graph(G)
    G2 = G.to_undirected()
    return G2


def clean_sub_graphs(G_,
                     min_length=150,
                     max_nodes_to_skip=30,
                     weight='length_pix',
                     verbose=True,
                     super_verbose=False):
    '''Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length
       (this step great improves processing time)'''

    if len(list(G_.nodes())) == 0:
        return G_

    if verbose:
        print ("Running clean_sub_graphs...")
    sub_graphs = list(nx.connected_component_subgraphs(G_))
    bad_nodes = []
    if verbose:
        print (" len(G_.nodes()):", len(list(G_.nodes())) )
        print (" len(G_.edges()):", len(list(G_.edges())) )
    if super_verbose:
        print ("G_.nodes:", G_.nodes())
        edge_tmp = G_.edges()[np.random.randint(len(G_.edges()))]
        print (edge_tmp, "G.edge props:", G_.edge[edge_tmp[0]][edge_tmp[1]])

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue

        else:
            all_lengths = dict(nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            if super_verbose:
                        print ("  \nGs.nodes:", G_sub.nodes() )
                        print ("  all_lengths:", all_lengths )
            # get all lenghts
            lens = []
            #for u,v in all_lengths.iteritems():
            for u in all_lengths.keys():
                v = all_lengths[u]
                #for uprime, vprime in v.iteritems():
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)
                    if super_verbose:
                        print ("  u, v", u,v )
                        print ("    uprime, vprime:", uprime, vprime )
            max_len = np.max(lens)
            if super_verbose:
                print ("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                if super_verbose:
                    print (" appending to bad_nodes:", G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print (" num bad_nodes:", len(bad_nodes))
        #print ("bad_nodes:", bad_nodes)
        print (" len(G'.nodes()):", len(G_.nodes()))
        print (" len(G'.edges()):", len(G_.edges()))
    if super_verbose:
        print ("  G_.nodes:", G_.nodes())

    return G_


def pixelToGeoCoord(xPix,
                    yPix,
                    inputRaster,
                    sourceSR='',
                    geomTransform='',
                    targetSR=''):
    '''from spacenet geotools'''
    # If you want to garuntee lon lat output, specify TargetSR  otherwise, geocoords will be in image geo reference
    # targetSR = osr.SpatialReference()
    # targetSR.ImportFromEPSG(4326)
    # Transform can be performed at the polygon level instead of pixel level

    if targetSR =='':
        performReprojection=False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection=True

    if geomTransform=='':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())

    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

    xCoord = (xPix * pixelWidth) + xOrigin
    yCoord = (yPix * pixelHeight) + yOrigin
    geom.AddPoint(xCoord, yCoord)


    if performReprojection:
        if sourceSR=='':
            srcRaster = gdal.Open(inputRaster)
            sourceSR = osr.SpatialReference()
            sourceSR.ImportFromWkt(srcRaster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(sourceSR, targetSR)
        geom.Transform(coord_trans)

    return (geom.GetX(), geom.GetY())


def get_node_geo_coords(G, im_file, verbose=False):
    nn = len(G.nodes())
    for i,(n,attr_dict) in enumerate(G.nodes(data=True)):
        if verbose:
            print ("node:", n)
        if (i % 1000) == 0:
            if verbose:
                print ("node", i, "/", nn)
        x_pix, y_pix = attr_dict['x_pix'], attr_dict['y_pix']
        lon, lat = pixelToGeoCoord(x_pix, y_pix, im_file)
        [utm_east, utm_north, utm_zone, utm_letter] =\
                    utm.from_latlon(lat, lon)
        attr_dict['lon'] = lon
        attr_dict['lat'] = lat
        attr_dict['utm_east'] = utm_east
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        attr_dict['utm_north'] = utm_north
        attr_dict['x'] = lon
        attr_dict['y'] = lat
        if verbose:
            print (" ", n, attr_dict)

    return G


def convert_pix_lstring_to_geo(wkt_lstring, im_file):
    '''Convert linestring in pixel coords to geo coords'''
    shape = wkt_lstring  #shapely.wkt.loads(lstring)
    x_pixs, y_pixs = shape.coords.xy
    coords_latlon = []
    coords_utm = []
    for (x,y) in zip (x_pixs, y_pixs):
        lon, lat = pixelToGeoCoord(x, y, im_file)
        [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)
        coords_utm.append([utm_east, utm_north])
        coords_latlon.append([lon, lat])

    lstring_latlon = LineString([Point(z) for z in coords_latlon])
    lstring_utm = LineString([Point(z) for z in coords_utm])

    return lstring_latlon, lstring_utm, utm_zone, utm_letter


def get_edge_geo_coords(G,
                        im_file,
                        remove_pix_geom=True,
                        verbose=False):
    ne = len(list(G.edges()))
    for i,(u,v,attr_dict) in enumerate(G.edges(data=True)):
        if verbose:
            print ("edge:", u,v)
        if (i % 1000) == 0:
            if verbose:
                print ("edge", i, "/", ne)
        geom_pix = attr_dict['geometry_pix']
        lstring_latlon, lstring_utm, utm_zone, utm_letter = convert_pix_lstring_to_geo(geom_pix, im_file)
        attr_dict['geometry_latlon_wkt'] = lstring_latlon.wkt
        attr_dict['geometry_utm_wkt'] = lstring_utm.wkt
        attr_dict['length_latlon'] = lstring_latlon.length
        attr_dict['length_utm'] = lstring_utm.length
        attr_dict['length'] = lstring_utm.length
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        if verbose:
            print ("  attr_dict:", attr_dict)

        # geometry screws up osmnx.simplify function
        if remove_pix_geom:
            #attr_dict['geometry_wkt'] = lstring_latlon.wkt
            attr_dict['geometry_pix'] = geom_pix.wkt

    return G
