# Author: 赩林, xilin0x7f@163.com
import numpy as np
import nibabel as nib
import subprocess
from .surface_func import compute_surface_area

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
    cifti_infile_data = cifti_infile.get_fdata()
    cifti_label = nib.load(str(out_prefix) + '.threshold.label.dscalar.nii')
    cifti_label_data = cifti_label.get_fdata()
    atlas_file = nib.load(atlas_path)
    atlas_data = atlas_file.get_fdata()[0]
    atlas_header = atlas_file.header
    atlas_map = {k: v[0] for k, v in atlas_header.get_axis(0).label[0].items()}
    brain_models = [bm for bm in atlas_header.get_index_map(1).brain_models]

    surf_lh_vert = surf_lh.darrays[0].data
    surf_rh_vert = surf_rh.darrays[0].data

    len_infile_label_equal = True
    if len(cifti_infile_data) > len(cifti_label_data):
        len_infile_label_equal = False
        print("will use label file multi time")
    elif len(cifti_infile_data) < len(cifti_label_data):
        ValueError("cifti file time size is less than label file")

    for darray_idx in range(len(cifti_infile_data)):
        cluster_all = []
        cluster_size_info_all = []
        cifti_infile_data_c = cifti_infile_data[darray_idx]
        if len_infile_label_equal:
            cifti_label_data_c = cifti_label_data[darray_idx]
        else:
            cifti_label_data_c = cifti_label_data[0]
        for label_idx in range(np.int64(np.max(cifti_label_data_c))):
            if label_idx + 1 not in np.unique(cifti_label_data_c):
                continue

            if less_than:
                peak_idx = np.array(range(len(cifti_label_data_c)))[cifti_label_data_c == (label_idx + 1)][np.argmin(
                    cifti_infile_data_c[cifti_label_data_c == (label_idx + 1)])]

            else:
                peak_idx = np.array(range(len(cifti_label_data_c)))[cifti_label_data_c == (label_idx + 1)][np.argmax(
                    cifti_infile_data_c[cifti_label_data_c == (label_idx + 1)])]

            peak_value = cifti_infile_data_c[peak_idx]
            peak_bm = None
            for bm in brain_models:
                start = bm.index_offset
                end = start + bm.index_count
                if start <= peak_idx < end:
                    peak_bm = bm

            if peak_bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT':
                peak_coord = surf_lh_vert[peak_bm._vertex_indices._indices][peak_idx - peak_bm.index_offset]
                cluster_size = np.sum(area_lh[peak_bm._vertex_indices._indices][
                    (cifti_label_data_c == (label_idx + 1))
                    [peak_bm.index_offset:peak_bm.index_offset + peak_bm.index_count]
                                      ])

                for inter_label_idx in np.unique(atlas_data[cifti_label_data_c == (label_idx + 1)]):
                    cluster_size_info_all.append([
                        label_idx + 1, atlas_map[inter_label_idx],
                        np.sum(area_lh[peak_bm._vertex_indices._indices][
                                   ((cifti_label_data_c == (label_idx + 1)) & (atlas_data == inter_label_idx))
                                   [peak_bm.index_offset:peak_bm.index_offset + peak_bm.index_count]
                               ]),
                        ])
            elif peak_bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
                peak_coord = surf_rh_vert[peak_bm._vertex_indices._indices][peak_idx - peak_bm.index_offset]
                cluster_size = np.sum(area_rh[peak_bm._vertex_indices._indices][
                                          (cifti_label_data_c == (label_idx + 1))
                                          [peak_bm.index_offset:peak_bm.index_offset + peak_bm.index_count]
                                      ])

                for inter_label_idx in np.unique(atlas_data[cifti_label_data_c == (label_idx + 1)]):
                    cluster_size_info_all.append([
                        label_idx + 1, atlas_map[inter_label_idx],
                        np.sum(area_rh[peak_bm._vertex_indices._indices][
                                   ((cifti_label_data_c == (label_idx + 1)) & (atlas_data == inter_label_idx))
                                   [peak_bm.index_offset:peak_bm.index_offset + peak_bm.index_count]
                               ]),
                        ])
            elif peak_bm.model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                peak_coord = (
                                 atlas_header.get_axis(1)._affine @
                                 np.append(atlas_header.get_axis(1).voxel[peak_idx], 1)
                             )[:3]
                cluster_size = np.sum(cifti_label_data_c == (label_idx + 1))

                for inter_label_idx in np.unique(atlas_data[cifti_label_data_c == (label_idx + 1)]):
                    cluster_size_info_all.append([
                        label_idx + 1, atlas_map[inter_label_idx],
                        np.sum((cifti_label_data_c == (label_idx + 1)) & (atlas_data == inter_label_idx)),
                        ])
            else:
                raise ValueError(f"peak_brain_structure is not support, not CORTEX_LEFT or CORTEX_RIGHT or volume")

            peak_region = atlas_map[atlas_data[peak_idx]]

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
