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

def volume_create_sphere(volume_path, out_path, x, y, z, r):
    from scipy.spatial.distance import cdist

    nii_file = nib.load(volume_path)
    data = nii_file.get_fdata()
    affine = nii_file.affine
    coord = np.array((x, y, z)).reshape(1, -1)
    mask = np.zeros(data.shape[:3]).ravel()
    i, j, k = np.indices(data.shape[:3])
    coords = (affine @ np.vstack((i.ravel(), j.ravel(), k.ravel(), np.ones(mask.size))))[:3, :].T
    mask[cdist(coord, coords, metric='euclidean')[0] < r] = 1
    nib.save(nib.Nifti1Image(mask.reshape(data.shape[:3]), affine), out_path)
