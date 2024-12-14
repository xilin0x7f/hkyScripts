#!/bin/env python
# Author: 赩林, xilin0x7f@163.com
import argparse
import inspect

from hkypy.cifti_func import cifti_array2map
from hkypy.dwi_fit import dmri_amico_fit
from hkypy.surface_func import surface_get_parc_coord
from hkypy.volume_func import volume_fpca, volume_4d2rgb, volume_extract, volume_restore

def arg_extractor(func):
    def wrapper(args):
        func_args = {
            k: v for k, v in vars(args).items()
            if k in func.__code__.co_varnames
        }
        return func(**func_args)
    return wrapper

def setup_dmri_amico_fit(subparsers):
    parser = subparsers.add_parser("dmri-amico-fit", help="dmri-amico-fit")
    parser.set_defaults(func=arg_extractor(dmri_amico_fit))
    parser.add_argument("dwi_path")
    parser.add_argument("bvec_path")
    parser.add_argument("bval_path")
    parser.add_argument('mask_path')
    parser.add_argument('model_name', nargs="?", default="NODDI")

def setup_volume_extract(subparsers):
    parser = subparsers.add_parser("volume-extract", help="extract value from nifti file in mask")
    parser.set_defaults(func=arg_extractor(volume_extract))
    parser.add_argument("volume_path", help="volume path")
    parser.add_argument("mask_path", help="mask path")
    parser.add_argument("out_path", help="out path, txt format")

def setup_volume_restore(subparsers):
    parser = subparsers.add_parser("volume-restore", help="restore nifti file in mask")
    parser.set_defaults(func=arg_extractor(volume_restore))
    parser.add_argument("data_path", help="data path")
    parser.add_argument("mask_path", help="mask path")
    parser.add_argument("out_path", help="out path")

def setup_volume_fpca(subparsers):
    parser = subparsers.add_parser("volume-fpca", help="functional PCA")
    parser.set_defaults(func=arg_extractor(volume_fpca))
    parser.add_argument('volume_path', help='4d nifti file')
    parser.add_argument('atlas_path', help='atlas file')
    parser.add_argument('out_prefix', help='output prefix')

def setup_cifti_array2map(subparsers):
    parser = subparsers.add_parser("cifti-array2map", help="convert cifti atlas array to map")
    parser.set_defaults(func=arg_extractor(cifti_array2map))
    parser.add_argument('atlas_path', help='dlabel file')
    parser.add_argument('data_path', help='csv file, per row is a brain region')
    parser.add_argument('out_path', help='output dscalar file')
    parser.add_argument('-d', '--delimiter', default=' ', help='delimiter for txt file')
    parser.add_argument('-t', '--transpose', action='store_true', help='transpose data')
    parser.add_argument('-s', '--skiprows', type=int, default=0, help='skip first rows')

def setup_volume_4d2rgb(subparsers):
    parser = subparsers.add_parser("volume-4d2rgb", help="convert 4d nifti to rgb nifti")
    parser.set_defaults(func=arg_extractor(volume_4d2rgb))
    parser.add_argument("volume_path", help="volume path")
    parser.add_argument("out_path", help="out path")

def setup_surface_get_parc_coord(subparsers):
    parser = subparsers.add_parser("surface-get-parc-coord", help="get parcellation coordinate from surface")
    parser.set_defaults(func=arg_extractor(surface_get_parc_coord))
    parser.add_argument("surf_path", help="surface path")
    parser.add_argument("parc_path", help="parcellation path")
    parser.add_argument("out_path", help="out path")

def main():
    parser = argparse.ArgumentParser(description="赩林")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # Dynamically find and register all setup_* functions
    current_module = inspect.getmembers(inspect.getmodule(main), inspect.isfunction)
    setup_functions = [func for name, func in current_module if name.startswith("setup_")]
    for setup_func in setup_functions:
        setup_func(subparsers)

    args = parser.parse_args()
    # print(vars(args))
    args.func(args)


if __name__ == "__main__":
    main()
