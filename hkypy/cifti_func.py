# Author: 赩林, xilin0x7f@163.com
import numpy as np
import nibabel as nib
import subprocess
from .surface_func import compute_surface_area

def brain_models_are_equal(cifti_header1, cifti_header2):
    brain_models1 = list(cifti_header1.get_index_map(1).brain_models)
    brain_models2 = list(cifti_header2.get_index_map(1).brain_models)

    if len(brain_models1) != len(brain_models2):
        return False

    for bm1, bm2 in zip(brain_models1, brain_models2):
        if bm1.model_type == "CIFTI_MODEL_TYPE_SURFACE":
            if (bm1.brain_structure != bm2.brain_structure or
                bm1.index_offset != bm2.index_offset or
                bm1.index_count != bm2.index_count or
                not (bm1._vertex_indices._indices == bm2._vertex_indices._indices)
            ):
                return False
        elif bm1.model_type == "CIFTI_MODEL_TYPE_VOXELS":
            if (bm1.brain_structure != bm2.brain_structure or
                bm1.index_offset != bm2.index_offset or
                bm1.index_count != bm2.index_count or
                not (bm1._voxel_indices_ijk._indices == bm2._voxel_indices_ijk._indices)
            ):
                return False

    return True

def cifti_array2map(atlas_path, data_path, out_path, delimiter=' ', transpose=False, skiprows=0):
    atlas = nib.load(atlas_path)
    data = np.loadtxt(data_path, delimiter=delimiter, skiprows=skiprows)
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
    nib.save(new_img, out_path)

def cifti_find_clusters(
    cifti_path, surface_value_threshold, surface_minimum_area, volume_value_threshold, volume_minimum_size,
    left_surface_path, right_surface_path, out_prefix, less_than=False, direction='COLUMN'
):
    command = [
        'wb_command', '-cifti-find-clusters', cifti_path, surface_value_threshold, surface_minimum_area,
        volume_value_threshold, volume_minimum_size, direction, str(out_prefix) + '.threshold.label.dscalar.nii'
    ]

    if less_than:
        command.append('-less-than')

    command.extend(['-left-surface', left_surface_path[0]])
    if len(left_surface_path) > 1:
        command.extend(['-corrected-areas', left_surface_path[1]])

    command.extend(['-right-surface', right_surface_path[0]])
    if len(right_surface_path) > 1:
        command.extend(['-corrected-areas', right_surface_path[1]])

    command = [str(i) for i in command]

    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)

    if str(result.stderr) != '':
        print(result.stderr)
        ChildProcessError('wb_command failed')

    command = [
        'wb_command',
        '-cifti-math',
        'x*(y>0)',
        str(out_prefix) + '.threshold.dscalar.nii',
        '-var', 'x', cifti_path,
        '-var', 'y', str(out_prefix) + '.threshold.label.dscalar.nii'
    ]
    command = [str(i) for i in command]

    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)

    if str(result.stderr) != '':
        print(result.stderr)
        ChildProcessError('wb_command failed')

def cifti_report(
    cifti_path, surface_value_threshold, surface_minimum_area, volume_value_threshold, volume_minimum_size, atlas_path,
    left_surface_path, right_surface_path, out_prefix, less_than=False, direction='COLUMN'
):
    import pandas as pd
    cifti_find_clusters(
        cifti_path, surface_value_threshold, surface_minimum_area, volume_value_threshold, volume_minimum_size,
        left_surface_path, right_surface_path, out_prefix, less_than=less_than, direction=direction
    )
    surf_lh = nib.load(left_surface_path[0])
    if len(left_surface_path) == 1:
        vertices_surface, faces_surface = surf_lh.darrays[0].data, surf_lh.darrays[1].data
        area_lh = compute_surface_area(vertices_surface, faces_surface)
    else:
        area_lh = nib.load(left_surface_path[1]).darrays[0].data

    surf_rh = nib.load(right_surface_path[0])
    if len(right_surface_path) == 1:
        vertices_surface, faces_surface = surf_rh.darrays[0].data, surf_rh.darrays[1].data
        area_rh = compute_surface_area(vertices_surface, faces_surface)
    else:
        area_rh = nib.load(right_surface_path[1]).darrays[0].data

    cifti_infile = nib.load(cifti_path)
    cifti_infile_brain_models = list(cifti_infile.header.matrix.get_index_map(1).brain_models)

    cifti_infile_data = cifti_infile.get_fdata()
    cifti_label = nib.load(str(out_prefix) + '.threshold.label.dscalar.nii')
    cifti_label_data = cifti_label.get_fdata()
    atlas_file = nib.load(atlas_path)

    atlas_file_brain_models = list(atlas_file.header.matrix.get_index_map(1).brain_models)

    # if not brain_models_are_equal(cifti_infile.header, atlas_file.header):
    #     raise ValueError("barin models of cifti file and atlas file are not equal")

    atlas_data = atlas_file.get_fdata()[0]
    atlas_header = atlas_file.header
    atlas_map = {k: v[0] for k, v in atlas_header.get_axis(0).label[0].items()}

    surf_lh_vert = surf_lh.darrays[0].data
    surf_rh_vert = surf_rh.darrays[0].data

    len_infile_label_equal = True
    if len(cifti_infile_data) > len(cifti_label_data):
        len_infile_label_equal = False
        print("will use label file multi time")
    elif len(cifti_infile_data) < len(cifti_label_data):
        ValueError("cifti file time size is less than label file")

    cifti_infile_lh, cifti_infile_rh, _, _, cifti_infile_volume, cifti_infile_volume_mat = cifti_separate(cifti_infile)
    cifti_label_lh, cifti_label_rh, _, _, cifti_label_volume, cifti_label_volume_mat = cifti_separate(cifti_label)
    atlas_file_lh, atlas_file_rh, _, _, atlas_file_volume, atlas_file_volume_mat = cifti_separate(atlas_file)
    if cifti_infile_lh.shape[1] != atlas_file_lh.shape[1]:
        raise ValueError("left surface and atlas file are not equal")
    if cifti_infile_rh.shape[1] != atlas_file_rh.shape[1]:
        raise ValueError("right surface and atlas file are not equal")

    volume_report = True
    if cifti_infile_volume is None:
        volume_report = False
    if atlas_file_volume is None:
        volume_report = False

    if volume_report and not (
        np.all(cifti_infile_volume.shape[:3] == atlas_file_volume.shape[:3]) and
        np.all(cifti_infile_volume_mat == cifti_infile_volume_mat)
            ):
        raise ValueError('volume matrix is not equal between cifti file and atlas file')

    for darray_idx in range(len(cifti_infile_data)):
        cluster_all = []
        cluster_size_info_all = []
        if len_infile_label_equal:
            cifti_label_data_c = cifti_label_data[darray_idx]
        else:
            cifti_label_data_c = cifti_label_data[0]
        for label_idx in range(np.int64(np.max(cifti_label_data_c))):
            if label_idx + 1 not in np.unique(cifti_label_data_c):
                continue

            if less_than:
                peak_idx = np.array(range(len(cifti_label_data_c)))[cifti_label_data_c == (label_idx + 1)][np.argmin(
                    cifti_infile_data[darray_idx][cifti_label_data_c == (label_idx + 1)])]

            else:
                peak_idx = np.array(range(len(cifti_label_data_c)))[cifti_label_data_c == (label_idx + 1)][np.argmax(
                    cifti_infile_data[darray_idx][cifti_label_data_c == (label_idx + 1)])]

            peak_value = cifti_infile_data[darray_idx][peak_idx]
            peak_bm = None
            for bm in cifti_infile_brain_models:
                start = bm.index_offset
                end = start + bm.index_count
                if start <= peak_idx < end:
                    peak_bm = bm

            if peak_bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT':
                peak_coord = surf_lh_vert[peak_bm._vertex_indices._indices][peak_idx - peak_bm.index_offset]
                peak_region = atlas_map[
                    atlas_file_lh[0][peak_bm._vertex_indices._indices][peak_idx - peak_bm.index_offset]
                ]
                cluster_size = np.sum(area_lh[peak_bm._vertex_indices._indices][
                    (cifti_label_data_c == (label_idx + 1))
                    [peak_bm.index_offset:peak_bm.index_offset + peak_bm.index_count]
                                      ])
                for inter_label_idx in np.unique(
                    atlas_file_lh[0][
                        (cifti_label_lh[darray_idx] if len_infile_label_equal else cifti_label_lh[0]) == (label_idx + 1)
                    ]
                ):
                    cluster_size_info_all.append([
                        label_idx + 1, atlas_map[inter_label_idx],
                        np.sum(area_lh[
                                   ((cifti_label_lh[darray_idx] if len_infile_label_equal else cifti_label_lh[0])
                                    == (label_idx + 1)) & (atlas_file_lh[0] == inter_label_idx)
                               ]),
                        ])
            elif peak_bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
                peak_coord = surf_rh_vert[peak_bm._vertex_indices._indices][peak_idx - peak_bm.index_offset]
                peak_region = atlas_map[
                    atlas_file_rh[0][peak_bm._vertex_indices._indices][peak_idx - peak_bm.index_offset]
                ]
                cluster_size = np.sum(area_rh[peak_bm._vertex_indices._indices][
                                          (cifti_label_data_c == (label_idx + 1))
                                          [peak_bm.index_offset:peak_bm.index_offset + peak_bm.index_count]
                                      ])
                for inter_label_idx in np.unique(
                    atlas_file_rh[0][
                        (cifti_label_rh[darray_idx] if len_infile_label_equal else cifti_label_rh[0]) == (label_idx + 1)
                    ]
                ):
                    cluster_size_info_all.append([
                        label_idx + 1, atlas_map[inter_label_idx],
                        np.sum(area_rh[
                                   ((cifti_label_rh[darray_idx] if len_infile_label_equal else cifti_label_rh[0])
                                    == (label_idx + 1)) & (atlas_file_rh[0] == inter_label_idx)
                                   ]),
                        ])
            elif peak_bm.model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                peak_coord = (
                                 cifti_infile.header.get_axis(1)._affine @
                                 np.append(cifti_infile.header.get_axis(1).voxel[peak_idx], 1)
                             )[:3]
                cluster_size = np.sum(cifti_label_data_c == (label_idx + 1))
                if volume_report:
                    peak_region = atlas_map[
                        atlas_file_volume[*peak_bm._voxel_indices_ijk._indices[peak_idx - peak_bm.index_offset], 0]
                    ]
                    for inter_label_idx in np.unique(
                        atlas_file_volume[
                            (
                                cifti_label_volume[..., darray_idx] if len_infile_label_equal else
                                cifti_label_volume[..., 0]
                            ) ==
                            (label_idx + 1), 0]
                    ):
                        cluster_size_info_all.append([
                            label_idx + 1, atlas_map[inter_label_idx],
                            np.sum(((
                                       cifti_label_volume[..., darray_idx] if len_infile_label_equal else
                                       cifti_label_volume[..., 0]
                                   ) ==
                                   (label_idx + 1)) &
                                   (atlas_file_volume[..., 0] == inter_label_idx)),
                            ])
                else:
                    peak_region = None
            else:
                raise ValueError(f"peak_brain_structure is not support, not CORTEX_LEFT or CORTEX_RIGHT or volume")

            cluster_all.append([label_idx+1, *peak_coord, peak_value, cluster_size, peak_region])

        cluster_all = pd.DataFrame(
            cluster_all,
            columns=['cluster idx', 'coord-x', 'coord-y', 'coord-z', 'peak value', 'size', 'annot']
        )

        cluster_all.to_csv(out_prefix + f'_darray-{darray_idx}_cluster_info.csv', index=False)
        cluster_size_info_all = pd.DataFrame(
            cluster_size_info_all,
            columns=['cluster idx', 'region', 'size']
        )
        cluster_size_info_all.to_csv(out_prefix + f'_darray-{darray_idx}_cluster_info_size.csv', index=False)

def cifti_separate(cifti_file):
    cifti_data = cifti_file.get_fdata()
    brain_models = cifti_file.header.matrix.get_index_map(1).brain_models

    volume_data, volume_mat = None, None
    ax1 = cifti_file.header.get_axis(1)
    if ax1._volume_shape is not None:
        volume_dim = ax1._volume_shape
        volume_mat = ax1._affine
        volume_data = np.zeros(list(volume_dim) + [cifti_data.shape[0]])

    cortex_left_data, cortex_left_indices = None, None
    cortex_right_data, cortex_right_indices = None, None

    for bm in brain_models:
        if bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
            cortex_left_data = np.zeros([cifti_data.shape[0], bm.surface_number_of_vertices])
            cortex_left_indices = bm._vertex_indices._indices
            cortex_left_data[:, cortex_left_indices] = cifti_data[:, bm.index_offset:bm.index_offset + bm.index_count]

        elif bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            cortex_right_data = np.zeros([cifti_data.shape[0], bm.surface_number_of_vertices])
            cortex_right_indices = bm._vertex_indices._indices
            cortex_right_data[:, cortex_right_indices] = cifti_data[:, bm.index_offset:bm.index_offset + bm.index_count]

        elif bm.model_type == "CIFTI_MODEL_TYPE_VOXELS":
            indices = np.array(bm._voxel_indices_ijk._indices)
            volume_data[indices[:, 0], indices[:, 1], indices[:, 2], :] = cifti_data[:, bm.index_offset:bm.index_offset + bm.index_count].T

    return cortex_left_data, cortex_right_data, cortex_left_indices, cortex_right_indices, volume_data, volume_mat

def cifti_surface_zscore(cifti_path, out_path, mask_path=None, weight_path=None):
    cifti_file = nib.load(cifti_path)
    mask_file = nib.load(mask_path) if mask_path is not None else None
    weight_file = nib.load(weight_path) if weight_path is not None else None
    cortex_left_data, cortex_right_data, cortex_left_indices, cortex_right_indices = cifti_separate(cifti_file)[:4]
    if mask_file is not None:
        mask_left, mask_right = cifti_separate(mask_file)[0:2]
    else:
        mask_left = np.zeros(cortex_left_data.shape[1])
        mask_left[cortex_left_indices] = 1
        mask_right = np.zeros(cortex_right_data.shape[1])
        mask_right[cortex_right_indices] = 1

    mask_left, mask_right = mask_left.flatten(), mask_right.flatten()

    if weight_file is not None:
        weight_lh, weight_rh = cifti_separate(weight_file)[0:2]
        weight_lh, weight_rh = weight_lh.flatten(), weight_rh.flatten()
    else:
        weight_lh, weight_rh = np.ones(cortex_left_data.shape[1]), np.ones(cortex_right_data.shape[1])

    cortex_all = np.hstack([cortex_left_data, cortex_right_data])
    mask_all = np.hstack([mask_left, mask_right])
    cortex_masked = cortex_all[:, mask_all > 0]
    if weight_lh is not None and weight_rh is not None:
        weight_all = np.hstack([weight_lh, weight_rh])
        weight_masked = weight_all[mask_all > 0]
        mean_value = np.average(cortex_masked, weights=weight_masked, axis=1)[:, np.newaxis]
        std_value = np.sqrt(np.average((cortex_masked-mean_value)**2, weights=weight_masked, axis=1))[:, np.newaxis]
        cortex_masked_zscore = (cortex_masked - mean_value) / std_value
    else:
        cortex_masked_zscore = (cortex_masked - np.nanmean(cortex_masked, axis=1)[:, np.newaxis]) / np.nanstd(cortex_masked, axis=1)[:, np.newaxis]

    # cortex_all_zscore = cortex_all * 0
    # cortex_all_zscore[:, mask_all > 0] = cortex_masked_zscore

    mask_left_vertex = np.array(range(cortex_left_data.shape[1]))
    mask_left_vertex = mask_left_vertex[mask_left > 0]

    mask_right_vertex = np.array(range(cortex_right_data.shape[1]))
    mask_right_vertex = mask_right_vertex[mask_right > 0]

    # ax0 = cifti_file.header.get_axis(0)
    ax0 = nib.cifti2.cifti2_axes.ScalarAxis(name=[f"#{i+1}" for i in range(cortex_all.shape[0])])
    # ax1 = cifti_file.header.get_axis(1)
    ax1 = nib.cifti2.cifti2_axes.BrainModelAxis(
        name=np.array(['CIFTI_STRUCTURE_CORTEX_LEFT' for _ in range(np.sum(mask_left > 0))] +
                      ['CIFTI_STRUCTURE_CORTEX_RIGHT' for _ in range(np.sum(mask_right > 0))]),
        nvertices={
            "CIFTI_STRUCTURE_CORTEX_LEFT": cortex_left_data.shape[1],
            "CIFTI_STRUCTURE_CORTEX_RIGHT": cortex_right_data.shape[1]},
        vertex=np.hstack([mask_left_vertex, mask_right_vertex])
    )
    header = nib.Cifti2Header.from_axes((ax0, ax1))
    img = nib.Cifti2Image(cortex_masked_zscore, header)
    img.nifti_header.set_intent('ConnDenseScalar')

    nib.save(img, out_path)

def cifti_extract(cifti_path, atlas_path, out_path, weight_path=None):
    import pandas as pd
    cifti_file, atlas_file = nib.load(cifti_path), nib.load(atlas_path)
    weight_file = nib.load(weight_path) if weight_path is not None else None
    cf_lh, cf_rh, _, _, cf_vol, cf_mat = cifti_separate(cifti_file)
    al_lh, al_rh, _, _, al_vol, al_mat = cifti_separate(atlas_file)
    al_lh, al_rh = al_lh.flatten(), al_rh.flatten()
    if weight_path is not None:
        wg_lh, wg_rh, _, _, wg_vol, wg_mat = cifti_separate(weight_file)
        wg_lh, wg_rh = wg_lh.flatten(), wg_rh.flatten()
        cf_lh = cf_lh * wg_lh
        cf_rh = cf_rh * wg_rh
        result_lh = [
            np.sum(cf_lh[:, al_lh == idx], axis=1) / np.sum(wg_lh[al_lh == idx])
            for idx in np.setdiff1d(np.unique(al_lh), 0)
        ]
        result_rh = [
            np.sum(cf_rh[:, al_rh == idx], axis=1) / np.sum(wg_rh[al_rh == idx])
            for idx in np.setdiff1d(np.unique(al_rh), 0)
        ]
    else:
        result_lh = [
            np.mean(cf_lh[:, al_lh == idx], axis=1)
            for idx in np.setdiff1d(np.unique(al_lh), 0)
        ]
        result_rh = [
            np.mean(cf_rh[:, al_rh == idx], axis=1)
            for idx in np.setdiff1d(np.unique(al_rh), 0)
        ]

    result_lh_df = pd.DataFrame(np.array(result_lh).T, columns=['lh'+str(i) for i in np.setdiff1d(np.unique(al_lh), 0)])
    result_rh_df = pd.DataFrame(np.array(result_rh).T, columns=['rh'+str(i) for i in np.setdiff1d(np.unique(al_rh), 0)])

    if cf_vol is not None:
        # 一致性检查
        if not np.all(cf_vol.shape[:3] == al_vol.shape[:3]):
            raise RuntimeError("cifti and atlas volume must have same shape")

        if np.linalg.norm(cf_mat-al_mat) > 1e-6:
            raise RuntimeError("cifti and atlas volume mat must equal")

        # weight mean
        if weight_path is not None and wg_vol is not None and (not np.all(np.unique(wg_vol) == 0)):
            if not np.all(cf_vol.shape[:3] == wg_vol.shape[:3]):
                raise RuntimeError("cifti and weight volume must have same shape")

            if np.linalg.norm(cf_mat-wg_mat) > 1e-6:
                raise RuntimeError("cifti and weight volume mat must equal")

            wg_vol, al_vol = wg_vol.flatten(), al_vol.flatten()
            cf_vol = cf_vol.reshape(-1, cf_vol.shape[3]).T * wg_vol
            result_vol = [
                np.sum(cf_vol[:, al_vol == idx], axis=1) / np.sum(wg_vol[al_vol == idx])
                for idx in np.setdiff1d(np.unique(al_vol), 0)
            ]
        else:
            al_vol = al_vol.flatten()
            cf_vol = cf_vol.reshape(-1, cf_vol.shape[3]).T
            result_vol = [
                np.mean(cf_vol[:, al_vol == idx], axis=1)
                for idx in np.setdiff1d(np.unique(al_vol), 0)
            ]

        result_vol_df = pd.DataFrame(
            np.array(result_vol).T, columns=['vol'+str(i) for i in np.setdiff1d(np.unique(al_vol), 0)]
        )
        result_df = pd.concat([result_lh_df, result_rh_df, result_vol_df], axis=1)
    else:
        result_df = pd.concat([result_lh_df, result_rh_df], axis=1)

    result_df.to_csv(out_path, index=False)

def cifti_cosine_distances(cifti_path, out_path, masks_path, threshold=5, not_fisher=False, save_corr=False):
    from .utils import column_wise_corr, row_wise_threshold
    from sklearn.metrics.pairwise import cosine_distances

    cifti = nib.load(cifti_path)
    mask1 = nib.load(masks_path[0])
    if len(masks_path) > 1:
        mask2 = nib.load(masks_path[1])
    else:
        mask2 = mask1

    if not brain_models_are_equal(cifti.header, mask1.header) & brain_models_are_equal(cifti.header, mask2.header):
        raise ValueError('brain models are not equal')

    data1 = cifti.get_fdata()[:, mask1.get_fdata().flatten() > 0]
    data2 = cifti.get_fdata()[:, mask2.get_fdata().flatten() > 0]

    corr_matrix = column_wise_corr(data1, data2)
    corr_matrix = np.nan_to_num(corr_matrix)

    if save_corr:
        np.savetxt(out_path.replace('.txt', '_corr_matrix.txt'), corr_matrix)

    corr_matrix = row_wise_threshold(corr_matrix, threshold=threshold)
    corr_matrix = np.nan_to_num(corr_matrix)

    if not_fisher:
        corr_matrix = cosine_distances(corr_matrix)
    else:
        corr_matrix = cosine_distances(np.nan_to_num(np.arctanh(corr_matrix)))

    np.savetxt(out_path, corr_matrix)

def cifti_restore(data, mask_path, out_path, transpose=False):
    import os
    if isinstance(data, str) and os.path.exists(data):
        data = np.loadtxt(data)

    mask = nib.load(mask_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if transpose:
        data = data.T

    data_out = np.zeros([data.shape[0], mask.get_fdata().shape[1]])
    data_out[:] = np.nan

    data_out[:, mask.get_fdata().flatten() > 0] = data

    ax0 = nib.cifti2.cifti2_axes.ScalarAxis(name=[f"#{i+1}" for i in range(data.shape[0])])
    ax1 = mask.header.get_axis(1)
    header = nib.Cifti2Header.from_axes((ax0, ax1))
    img = nib.Cifti2Image(data_out, header)

    nib.save(img, out_path)
