# Author: 赩林, xilin0x7f@163.com
import numpy as np
import nibabel as nib

def volume_fpca(volume_path, atlas_path, out_prefix):
    """
    fpca 用于进行Functional pca分析，并保存均值、Vmat到本地以便后续应用, 以保存的文件为pca_mean.nii.gz,
    roi1的Vmat为pca_Vmat_00001.nii.gz为例，用fsl, workbench提取all_FA_skeletonised_s2, roi1的主成分
    fslmaths all_FA_skeletonised_s2 -sub pca_mean all_FA_skeletonised_s2_demeaned
    fslroi pca_Vmat_00001 roi1_pc1_vmat 0 1
    fslmaths all_FA_skeletonised_s2_demeaned -mul roi1_pc1_vmat roi1_pc1_res
    wb_command -volume-stats roi1_pc1_res.nii.gz -reduce SUM
    输出结果结果为roi1 pc1的主成分
    """
    data_file = nib.load(volume_path)
    atlas_file = nib.load(atlas_path)

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
    nib.save(nifti_image, f'{out_prefix}_mean.nii.gz')

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
        nib.save(nifti_image, f'{out_prefix}_Vmat_{int(atlas_idx):05}.nii.gz')

        principal_components = U * S
        # principal_components space * n_components, 每一行是一个被试，每一列是一个成分
        np.savetxt(f'{out_prefix}_PC_{int(atlas_idx):05}.txt', principal_components)
        explained_variance = S ** 2 / (U.shape[0] - 1)
        explained_variance_ratio = S ** 2 / sum(S ** 2)
        # singular value, explained_variance, explained_variance_ratio
        pca_info = np.vstack([S, explained_variance, explained_variance_ratio]).T
        np.savetxt(f'{out_prefix}_PCA_Info_{int(atlas_idx):05}.txt', pca_info)

def volume_4d2rgb(volume_path, out_path):
    nii_header = nib.load(volume_path)
    nii_data = nii_header.get_fdata()

    nii_data = (
        np.abs(nii_data) * 255 /
        np.max(np.abs([nii_data.max(), nii_data.min()]))
    ).astype(np.int8).copy().view(dtype=np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]))

    nii_img = nib.Nifti1Image(nii_data, nii_header.affine)
    nib.save(nii_img, out_path)

def volume_extract(volume_path, mask_path, out_path=None):
    volume_img = nib.load(volume_path)
    mask_img = nib.load(mask_path)
    if np.linalg.norm(volume_img.affine - mask_img.affine) > 1e-6:
        raise ValueError("data and mask do not have the same affine")
    volume_data = volume_img.get_fdata()
    mask_data = mask_img.get_fdata()
    if out_path is not None:
        np.savetxt(out_path, volume_data[mask_data > 0])

    return volume_data[mask_data > 0]

def volume_restore(data, mask_path, out_path):
    import os
    if isinstance(data, str) and os.path.exists(data):
        data = np.loadtxt(data)

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    if data.ndim == 2:
        img_data = np.zeros((*mask_data.shape, data.shape[1]))
        img_data[mask_data > 0, :] = data
    else:
        img_data = np.zeros(mask_data.shape)
        img_data[mask_data > 0] = data

    img_header = nib.Nifti1Image(img_data, header=mask_img.header, affine=mask_img.header.get_best_affine())
    nib.save(img_header, out_path)

def volume_create_sphere(volume_path, out_path, x, y, z, r, equal=False):
    from scipy.spatial.distance import cdist

    nii_file = nib.load(volume_path)
    data = nii_file.get_fdata()
    affine = nii_file.affine
    coord = np.array((x, y, z)).reshape(1, -1)
    mask = np.zeros(data.shape[:3]).ravel()
    i, j, k = np.indices(data.shape[:3])
    coords = (affine @ np.vstack((i.ravel(), j.ravel(), k.ravel(), np.ones(mask.size))))[:3, :].T
    if equal:
        mask[cdist(coord, coords, metric='euclidean')[0] <= r] = 1
    else:
        mask[cdist(coord, coords, metric='euclidean')[0] < r] = 1

    nib.save(nib.Nifti1Image(mask.reshape(data.shape[:3]), affine), out_path)

def volume_create_rectangle(volume_path, out_path, x, y, z, size, less=True):
    nii_file = nib.load(volume_path)
    data = nii_file.get_fdata()
    affine = nii_file.affine
    coord = np.array((x, y, z, 1))
    mask = np.zeros(data.shape[:3])
    i, j, k = (np.linalg.inv(affine) @ coord)[:3]
    if len(size) == 1:
        size = (size[0], size[0], size[0])

    if less:
        mask[
            np.ceil(i-size[0]/2).astype(int):np.floor(i+size[0]/2).astype(int),
            np.ceil(j-size[1]/2).astype(int):np.floor(j+size[1]/2).astype(int),
            np.ceil(k-size[2]/2).astype(int):np.floor(k+size[2]/2).astype(int)
        ] = 1
    else:
        mask[
            np.floor(i-size[0]/2).astype(int):np.ceil(i+size[0]/2).astype(int),
            np.floor(j-size[1]/2).astype(int):np.ceil(j+size[1]/2).astype(int),
            np.floor(k-size[2]/2).astype(int):np.ceil(k+size[2]/2).astype(int)
        ] = 1

    nib.save(nib.Nifti1Image(mask, affine), out_path)

def volume_frame_intensity_censoring(volume_path, mask_path, out_path, thresh=0.005):
    volume_file = nib.load(volume_path)
    mask_file = nib.load(mask_path)
    assert np.all(volume_file.affine == mask_file.affine)
    volume_data, mask_data = volume_file.get_fdata(), mask_file.get_fdata().astype(bool)
    while True:
        delta_signal = np.diff(volume_data, axis=3)

        rms_change = (
            np.sqrt(np.mean(delta_signal[mask_data, :] ** 2, axis=0)) /
            np.mean(volume_data[mask_data, :-1], axis=0)
        )

        del_idx = np.where(rms_change > thresh)[0]
        if len(del_idx) == 0:
            break

        del_idx = np.unique(np.concatenate([del_idx - 1, del_idx, del_idx + 1]))
        del_idx = del_idx[np.bitwise_and(del_idx != -1, del_idx < volume_data.shape[3])]
        if len(del_idx) == volume_data.shape[3]:
            print('will delete all frames')

        volume_data = np.delete(volume_data, del_idx, axis=3)
        print(f'Deleted {len(del_idx)} frames')

    volume_data = np.zeros_like(mask_data) if volume_data.size == 0 else volume_data
    nib.save(nib.Nifti1Image(volume_data, volume_file.affine), out_path)

def volume_reset_idx(volume_path, info_path, out_path):
    import pandas as pd
    nii_file = nib.load(volume_path)
    nii_info = pd.read_csv(info_path)
    nii_data = nii_file.get_fdata()

    coord_real = np.hstack([
        nii_info[['x', 'y', 'z']].to_numpy().reshape(-1, 3),
        np.ones(len(nii_info))[:, np.newaxis]
    ])

    coord_voxel = np.round(
        (np.linalg.inv(nii_file.affine) @ coord_real.T).T[:, :3]
    ).astype(int)

    if np.min(nii_data) < 0:
        raise ValueError('Nii data must be non-negative')

    nii_data = -1 * nii_data
    for idx in range(len(nii_info)):
        nii_data[nii_data == nii_data[*coord_voxel[idx]]] = nii_info['cluster_idx'][idx]

    if np.min(nii_data) < 0:
        print('some clusters were not assigned, will remove them')
        nii_data[nii_data < 0] = 0

    nib.save(nib.Nifti1Image(nii_data, nii_file.affine), out_path)

def mni152_to_fsaverage(volume_path, out_prefix, density='164k', method='linear'):
    from neuromaps.transforms import mni152_to_fsaverage
    res_lh, res_rh = mni152_to_fsaverage(volume_path, density, method)
    nib.save(res_lh, f'{out_prefix}lh.func.gii')
    nib.save(res_rh, f'{out_prefix}rh.func.gii')

def mni152_to_fslr(volume_path, out_prefix, density='32k', method='linear'):
    from neuromaps.transforms import mni152_to_fslr
    res_lh, res_rh = mni152_to_fslr(volume_path, density, method)
    nib.save(res_lh, f'{out_prefix}lh.func.gii')
    nib.save(res_rh, f'{out_prefix}rh.func.gii')

def mni152_to_civet(volume_path, out_prefix, density='41k', method='linear'):
    from neuromaps.transforms import mni152_to_civet
    res_lh, res_rh = mni152_to_civet(volume_path, density, method)
    nib.save(res_lh, f'{out_prefix}lh.func.gii')
    nib.save(res_rh, f'{out_prefix}rh.func.gii')
