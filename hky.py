#!/bin/env python
# Author: 赩林, xilin0x7f@163.com
import argparse
import inspect

from hkypy.cifti_func import cifti_array2map
from hkypy.dwi_fit import dmri_amico_fit, dmri_dki_fit
from hkypy.surface_func import surface_get_parc_coord
from hkypy.volume_func import (
    volume_fpca, volume_4d2rgb, volume_extract, volume_restore, volume_create_sphere,
    volume_frame_intensity_censoring
)
from hkypy.combat import text_combat, volume_combat
from hkypy.metric_func import metric_report
from hkypy.utils import get_motion, get_matrix_tril, matrix_tril_to_matrix
from hkypy.cifti_func import cifti_report, cifti_surface_zscore

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
    parser.add_argument("data", help="data path or data")
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
    parser.add_argument('-s', '--skiprows', type=int, default=0, help='skip first rows, default is 0')

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

def setup_text_combat(subparsers):
    parser = subparsers.add_parser("text-combat", help="text combat")
    parser.set_defaults(func=arg_extractor(text_combat))
    parser.add_argument("text_path", help="data text_path")
    parser.add_argument("subj_info_path", help="subj info csv path")
    parser.add_argument("out_path", help="output data after combat path")
    parser.add_argument('-b', '--batch', default='site', help='site var name')
    parser.add_argument('-con', '--con', default=None, help='continuous')
    parser.add_argument('-cat', '--cat', default=None, help='categorical')
    parser.add_argument('-n', '--neb', help='-n not use eb; else use eb', action='store_false', dest='eb')
    parser.add_argument('-f', '--factor', default=0, help='factor, set 0 to auto estimate, default is 0', type=int)

def setup_volume_combat(subparsers):
    parser = subparsers.add_parser("volume-combat", help="volume combat")
    parser.set_defaults(func=arg_extractor(volume_combat))
    parser.add_argument("volume_path", help="data volume_path")
    parser.add_argument("mask_path", help="mask volume_path")
    parser.add_argument("subj_info_path", help="subj info csv path")
    parser.add_argument("out_path", help="output data after combat path")
    parser.add_argument('-b', '--batch', default='site', help='site var name')
    parser.add_argument('-con', '--con', default=None, help='continuous')
    parser.add_argument('-cat', '--cat', default=None, help='categorical')
    parser.add_argument('-n', '--neb', help='-n not use eb; else use eb', action='store_false', dest='eb')
    parser.add_argument('-f', '--factor', default=0, help='factor, set 0 (default) to auto estimate', type=int)

def setup_metric_report(subparsers):
    parser = subparsers.add_parser("metric-report", help="""
    metric report, example:
    python hky.py metric-report a_lh.func.gii -o a_lh -t 5 -a data/Desikan.32k.L.label.gii \
        -s data/tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii data/tpl-fsLR_den-32k_hemi-L_midthickness.area.shape.gii
    
    python hky.py metric-report a_rh.func.gii -o a_rh -t 5 -a data/Desikan.32k.R.label.gii \
        -s data/tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii data/tpl-fsLR_den-32k_hemi-R_midthickness.area.shape.gii
    """)
    parser.set_defaults(func=arg_extractor(metric_report))
    parser.add_argument('metric_path', help='input a metric file for report')
    parser.add_argument('-s', '--surface', required=True, nargs='+',
                        help='input a surface file [and a surface area file]', dest='surface_path')
    parser.add_argument('-a', '--atlas', required=True, help='input a atlas file for report brain regions',
                        dest='atlas_path')
    parser.add_argument('-o', '--out_prefix', required=True, help='out prefix')
    parser.add_argument('-t', '--threshold', required=False, default=1,
                        help='value-threshold for wb_command, default 1')
    parser.add_argument('--minimum-area', required=False, default=0, help='minimum area for wb_command, default 0')
    parser.add_argument('-less-than', action='store_true')

def setup_get_motion(subparsers):
    parser = subparsers.add_parser("get-motion", help="""
    get motion from xcpd results, example: hky.py get-motion motion.csv sub-*/ses-*/func/*_task-rest_motion.tsv
    """)
    parser.set_defaults(func=arg_extractor(get_motion))
    parser.add_argument("output_path", help="output path")
    parser.add_argument("files_path", nargs="+", help="motion files path")

def setup_cifti_report(subparsers):
    parser = subparsers.add_parser("cifti-report", help="""
    cifti report, example: find value > 5, area > 0, surface cluster. add -less-than flag if you want less than a value.
    python hky.py cifti-report a.dscalar.nii 5 0 5 0 data/Desikan.32k.dlabel.nii out_prefix \
        -l data/tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii data/tpl-fsLR_den-32k_hemi-L_midthickness.area.shape.gii \
        -r data/tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii data/tpl-fsLR_den-32k_hemi-R_midthickness.area.shape.gii
    """)
    parser.set_defaults(func=arg_extractor(cifti_report))
    parser.add_argument("cifti_path", help="cifti path")
    parser.add_argument("surface_value_threshold", help="surface value threshold")
    parser.add_argument("surface_minimum_area", help="surface minimum area")
    parser.add_argument("volume_value_threshold", help="volume value threshold")
    parser.add_argument("volume_minimum_size", help="volume minimum size")
    parser.add_argument("atlas_path", help="atlas path")
    parser.add_argument("out_prefix", help="out prefix")
    parser.add_argument("-l", "--left_surface_path", nargs="+", help="left surface path", required=True)
    parser.add_argument("-r", "--right_surface_path", nargs="+", help="right surface path", required=True)
    parser.add_argument('-less-than', action='store_true')
    parser.add_argument('-d', '--direction', default='COLUMN', help='COLUMN or ROW, default COLUMN')

def setup_cifti_surface_zscore(subparsers):
    parser = subparsers.add_parser("cifti-surface-zscore", help="""
    cifti surface zscore, example:
    python hky.py cifti-surface-zscore a.dscalar.nii a.zscore.dscalar.nii -m data/tpl-fsLR_den-32k_mask.dscalar.nii \
        -w data/tpl-fsLR_den-32k_midthickness.area.dscalar.nii
    """)
    parser.set_defaults(func=arg_extractor(cifti_surface_zscore))
    parser.add_argument("cifti_path", help="cifti path")
    parser.add_argument("out_path", help="out path")
    parser.add_argument("-m", "--mask_path", help="mask path")
    parser.add_argument("-w", "--weight_path", help="weight path")

def setup_dmri_dki_fit(subparsers):
    parser = subparsers.add_parser("dmri-dki-fit", help='dmri dki fit')
    parser.set_defaults(func=arg_extractor(dmri_dki_fit))
    parser.add_argument("dwi_path", help="dmri path")
    parser.add_argument('bvec_path', help='bvec path')
    parser.add_argument('bval_path', help='bval path')
    parser.add_argument("mask_path", help="mask path")
    parser.add_argument("out_prefix", default='dki_', help="out prefix, default dki")
    parser.add_argument('-f', '--fwhm', default=1.25, type=float,
                        help='fwhm for smooth, set 0 to no smooth, default 1.25')
    parser.add_argument('-m', '--method', default='WLS', help="""
    dki fit method, default WLS, support OLS, ULLS, WLS, WLLS, UWLLS, CLS, CWLS
    """)
    parser.add_argument('--min-kurtosis', default=0.0, type=float, help='minimum kurtosis, default 0.0')
    parser.add_argument('--max-kurtosis', default=3.0, type=float, help='maximum kurtosis, default 3.0')

def setup_volume_create_sphere(subparsers):
    parser = subparsers.add_parser('volume-create-sphere', help='create volume sphere mask by mm coord')
    parser.set_defaults(func=arg_extractor(volume_create_sphere))
    parser.add_argument("volume_path", help="volume path")
    parser.add_argument("out_path", help="out path")
    parser.add_argument("x", type=float, help='x mm coord')
    parser.add_argument("y", type=float, help='y mm coord')
    parser.add_argument("z", type=float, help='z mm coord')
    parser.add_argument("r", type=float, help="radius")

def setup_volume_frame_intensity_censoring(subparsers):
    parser = subparsers.add_parser('volume-frame-intensity-censoring', help='volume frame intensity censoring')
    parser.set_defaults(func=arg_extractor(volume_frame_intensity_censoring))
    parser.add_argument("volume_path", help="volume path")
    parser.add_argument("mask_path", help="mask path")
    parser.add_argument("out_path", help="out path")
    parser.add_argument("thresh", type=float, help='threshold value')

def setup_get_matrix_tril(subparsers):
    parser = subparsers.add_parser('get-matrix-tril', help="""
    get lower triangular of symmetric matrix to text file, example:
    hky.py get-matrix-tril [-diag] res.txt a.txt b.txt c.txt
    """)
    parser.set_defaults(func=arg_extractor(get_matrix_tril))
    parser.add_argument('-diag', action='store_true',
                        help='-diag to get diag of matrix, default is not diag, i.e. off diagonal'
                        )
    parser.add_argument("out_path", help="out path")
    parser.add_argument("matrix_paths", nargs='+', help="matrix path")

def setup_matrix_tril_to_matrix(subparsers):
    parser = subparsers.add_parser('matrix-tril-to-matrix', help="""
    convert lower triangular of symmetric matrix from text file to matrix files, example:
    hky.py matrix-tril-to-matrix [-diag] res.txt output_prefix
    """)
    parser.set_defaults(func=arg_extractor(matrix_tril_to_matrix))
    parser.add_argument('-diag', action='store_true',
                        help='-diag to get diag of matrix, default is not diag, i.e. off diagonal')
    parser.add_argument("tril_path", help="tril path")
    parser.add_argument("out_prefix", default='matrix', help="out prefix, default matrix")

def main():
    parser = argparse.ArgumentParser(description="Author: 赩林, Email: xilin0x7f@163.com")
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
