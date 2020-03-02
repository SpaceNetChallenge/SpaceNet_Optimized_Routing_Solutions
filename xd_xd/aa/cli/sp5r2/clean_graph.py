from pathlib import Path
import re
import json
import sys

import click

sys.path.append('./')  # aa

from aa.road_networks.build_coords import main as build_coords
from aa.road_networks.postproc_graph import main as postproc_stage1
from aa.road_networks.postproc_graph import main_stage2 as postproc_stage2


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    help_option_names=[],
))
@click.option('--input-file', type=str)
@click.option('--output-file', type=str)
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def main(input_file, output_file, args):
    args = ",".join(list(args)[1:])
    aoi_data_path_mapping = parse_paths(args)
    print(json.dumps(aoi_data_path_mapping, indent=4))

    fn_input_debug = input_file
    fn_out = output_file + '_stg1.csv'
    fn_out_debug = output_file + '_stg1_debug.csv'
    fn_out_stg2 = output_file
    fn_out_debug_stg2 = output_file + '_stg2_debug.csv'

    df_chiplocations = build_coords(aoi_data_path_mapping)
    postproc_stage1(fn_input_debug,
                    fn_out,
                    fn_out_debug,
                    df_chiplocations,
                    aoi_data_path_mapping)
    postproc_stage2(fn_out_debug,
                    fn_out_stg2,
                    fn_out_debug_stg2,
                    df_chiplocations,
                    aoi_data_path_mapping)


def parse_paths(args):
    path_mapping = {}
    for spacenet_folder in args.split(','):
        sample_image_files = list(Path(spacenet_folder).glob('./PS-MS/*_AOI_*.tif'))
        assert len(sample_image_files) > 0

        """
        The format of an image name is
        <prefix>_AOI_<n>_<city>_<type>_chip<i>.tif

        ref: https://www.topcoder.com/challenges/30099956?tab=details
        """
        m = re.match('.+_AOI_([^_]+)_(\w+)_([^_]+)_chip([^\.]+).tif', sample_image_files[0].name)
        assert m is not None

        aoi_n, city_name, type_name, chip_id = m.groups()
        aoi_name = f"AOI_{aoi_n}_{city_name}"
        path_mapping[aoi_name] = spacenet_folder

    return path_mapping


if __name__ == '__main__':
    main()
