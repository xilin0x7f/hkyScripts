# Author: 赩林, xilin0x7f@163.com
import argparse
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

@arg_extractor
def command3(input, output, i=10):
    print(f"Running command3 with input: {input} and output: {output}, i: {i}")

def main():
    parser = argparse.ArgumentParser(description="A script supporting multiple subcommands.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    parser1 = subparsers.add_parser("volume-extract", help="extract value from nifti file in mask")
    parser1.set_defaults(func=volume_extract)
    parser1.add_argument("volume_path", help="volume path")
    parser1.add_argument("mask_path", help="mask path")
    parser1.add_argument("save_path", help="save path, txt format")

    parser_2 = subparsers.add_parser("volume-restore", help="restore nifti file in mask")
    parser_2.set_defaults(func=volume_restore)
    parser_2.add_argument("data_path")
    parser_2.add_argument("mask_path")
    parser_2.add_argument("save_path")

    parser_command3 = subparsers.add_parser("command3", help="Run the second command")
    parser_command3.set_defaults(func=command3)
    parser_command3.add_argument("input", help="Input file for command2")
    parser_command3.add_argument("output", type=int, help="Parameter for command2")
    parser_command3.add_argument("-i", type=int, help="Parameter for command2")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
