#!/usr/bin/python
# Author: 赩林, xilin0x7f@163.com
import argparse
import inspect

import numpy as np
import nibabel as nib

def arg_extractor(func):
    def wrapper(args):
        func_args = {
            k: v for k, v in vars(args).items()
            if k in func.__code__.co_varnames
        }
        return func(**func_args)
    return wrapper

@arg_extractor
def volume_extract(volume_path, mask_path, save_path):
    volume_img = nib.load(volume_path)
    mask_img = nib.load(mask_path)
    volume_data = volume_img.get_fdata()
    mask_data = mask_img.get_fdata()
    np.savetxt(save_path, volume_data[mask_data > 0])

def setup_volume_extract(subparsers):
    parser = subparsers.add_parser("volume-extract", help="extract value from nifti file in mask")
    parser.set_defaults(func=volume_extract)
    parser.add_argument("volume_path", help="volume path")
    parser.add_argument("mask_path", help="mask path")
    parser.add_argument("save_path", help="save path, txt format")

@arg_extractor
def volume_restore(data_path, mask_path, save_path):
    data = np.loadtxt(data_path)
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    if data.ndim == 2:
        img_data = np.zeros((*mask_data.shape, data.shape[1]))
        img_data[mask_data > 0, :] = data
    else:
        img_data = np.zeros(mask_data.shape)
        img_data[mask_data > 0] = data

    img_header = nib.Nifti1Image(img_data, header=mask_img.header, affine=mask_img.header.get_best_affine())
    nib.save(img_header, save_path)

def setup_volume_restore(subparsers):
    parser = subparsers.add_parser("volume-restore", help="restore nifti file in mask")
    parser.set_defaults(func=volume_restore)
    parser.add_argument("data_path", help="data path")
    parser.add_argument("mask_path", help="mask path")
    parser.add_argument("save_path", help="save path")

@arg_extractor
def cifti_array2map(atlas, data, out, delimiter=' ', transpose=False, skiprows=0):
    atlas = nib.load(atlas)
    data = np.loadtxt(data, delimiter=delimiter, skiprows=skiprows)
    if transpose:
        data = data.T

    atlas_data = atlas.get_fdata()
    if np.ndim(data) == 1:
        new_data = np.zeros(atlas_data.shape)
        for i, value in enumerate(data):
            new_data[atlas_data == i + 1] = value
    else:
        new_data = np.zeros([data.shape[1], atlas_data.shape[1]])
        for i, value in enumerate(data):
            new_data[:, atlas_data[0, :] == i + 1] = value[:, np.newaxis]

    ax0 = nib.cifti2.cifti2_axes.ScalarAxis(name=[f"#{i+1}" for i in range(new_data.shape[0])])
    ax1 = atlas.header.get_axis(1)
    header = nib.Cifti2Header.from_axes((ax0, ax1))
    new_img = nib.Cifti2Image(new_data, header, atlas.nifti_header)
    nib.save(new_img, out)

def setup_cifti_array2map(subparsers):
    parser = subparsers.add_parser("cifti-array2map", help="convert cifti atlas array to map")
    parser.set_defaults(func=cifti_array2map)
    parser.add_argument('atlas', help='dlabel file')
    parser.add_argument('data', help='csv file, per row is a brain region')
    parser.add_argument('out', help='output dscalar file')
    parser.add_argument('-d', '--delimiter', default=' ', help='delimiter for txt file')
    parser.add_argument('-t', '--transpose', action='store_true', help='transpose data')
    parser.add_argument('-s', '--skiprows', type=int, default=0, help='skip first rows')

def main():
    parser = argparse.ArgumentParser(description="A script supporting multiple subcommands.")
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
