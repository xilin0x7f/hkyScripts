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
def fpca(data, atlas, out):
    """
    fpca 用于进行Functional pca分析，并保存均值、Vmat到本地以便后续应用, 以保存的文件为pca_mean.nii.gz,
    roi1的Vmat为pca_Vmat_00001.nii.gz为例，用fsl, workbench提取all_FA_skeletonised_s2, roi1的主成分
    fslmaths all_FA_skeletonised_s2 -sub pca_mean all_FA_skeletonised_s2_demeaned
    fslroi pca_Vmat_00001 roi1_pc1_vmat 0 1
    fslmaths all_FA_skeletonised_s2_demeaned -mul roi1_pc1_vmat roi1_pc1_res
    wb_command -volume-stats roi1_pc1_res.nii.gz -reduce SUM
    输出结果结果为roi1 pc1的主成分
    """
    data_file = nib.load(data)
    atlas_file = nib.load(atlas)

    if np.linalg.norm(data_file.affine - atlas_file.affine) > 1e-6:
        raise ValueError("data and atlas do not have the same affine")

    data = data_file.get_fdata()
    atlas = atlas_file.get_fdata()

    if not data.shape[:3] == atlas.shape:
        raise ValueError("data and atlas do not have the same shape")

    # time * space
    dat2d = data.reshape(-1, data.shape[3]).T
    dat2d_mean = np.mean(dat2d, axis=0)
    nifti_image = nib.Nifti1Image(
        dat2d_mean.reshape(data.shape[:3]), header=data_file.header, affine=data_file.header.get_best_affine()
    )
    nib.save(nifti_image, f'{out}_mean.nii.gz')

    dat2d = dat2d - dat2d_mean
    for atlas_idx in np.unique(atlas):
        if atlas_idx == 0:
            continue

        dat4pca = dat2d[:, atlas.flatten() == atlas_idx]
        U, S, Vh = np.linalg.svd(dat4pca, full_matrices=False)
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=dat4pca.shape[1])
        # principal_components = pca.fit_transform(dat4pca) # 此处的dat4pca可以是未减均值的
        # principal_components = dat4pca @ Vh.T = U * S
        # 以上三个方法计算的PC至少在前若干个主成分应该是相同的，为简便计，采用numpy svd进行计算
        # space * n_components
        dat2d4save = np.zeros((dat2d.shape[1], Vh.shape[0]))
        dat2d4save[atlas.flatten() == atlas_idx, :] = Vh.T
        dat4save = dat2d4save.reshape((*data.shape[:3], dat2d4save.shape[1]))
        nifti_image = nib.Nifti1Image(dat4save, header=data_file.header, affine=data_file.header.get_best_affine())
        nib.save(nifti_image, f'{out}_Vmat_{int(atlas_idx):05}.nii.gz')

        principal_components = U * S
        # principal_components space * n_components, 每一行是一个被试，每一列是一个成分
        np.savetxt(f'{out}_PC_{int(atlas_idx):05}.txt', principal_components)
        explained_variance = S ** 2 / (U.shape[0] - 1)
        explained_variance_ratio = S ** 2 / sum(S ** 2)
        # singular value, explained_variance, explained_variance_ratio
        pca_info = np.vstack([S, explained_variance, explained_variance_ratio]).T
        np.savetxt(f'{out}_PCA_Info_{int(atlas_idx):05}.txt', pca_info)

def setup_fpca(subparsers):
    parser = subparsers.add_parser("fpca", help="functional PCA")
    parser.set_defaults(func=fpca)
    parser.add_argument('data', help='4d nifti file')
    parser.add_argument('atlas', help='atlas file')
    parser.add_argument('out', help='output prefix')


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
