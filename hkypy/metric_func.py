# Author: 赩林, xilin0x7f@163.com
import numpy as np
import nibabel as nib
import subprocess
from .surface_func import compute_surface_area

def metric_find_clusters(metric_path, surface_path, threshold, minimum_area, output_prefix, less_than=False):
    command = [
        'wb_command',
        '-metric-find-clusters',
        surface_path[0],
        metric_path,
        threshold,
        minimum_area,
        str(output_prefix) + '.threshold.label.func.gii'
    ]

    if less_than:
        command.append('-less-than')

    if len(surface_path) > 1:
        command.extend(['-corrected-areas', surface_path[1]])

    command = [str(i) for i in command]

    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)

    if str(result.stderr) != '':
        print(result.stderr)
        ChildProcessError('wb_command failed')

    command = [
        'wb_command',
        '-metric-math',
        'x * (y >0)',
        str(output_prefix) + '.threshold.func.gii',
        '-var', 'x', metric_path,
        '-var', 'y', str(output_prefix) + '.threshold.label.func.gii'
    ]
    command = [str(i) for i in command]

    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)

    if str(result.stderr) != '':
        print(result.stderr)
        ChildProcessError('wb_command failed')


def metric_report(metric_path, surface_path, atlas_path, threshold, minimum_area, output_prefix, less_than=False):
    import pandas as pd
    metric_find_clusters(metric_path, surface_path, threshold, minimum_area, output_prefix, less_than)
    metric_surface = nib.load(surface_path[0])
    if len(surface_path) == 1:
        vertices_surface, faces_surface = metric_surface.darrays[0].data, metric_surface.darrays[1].data
        metric_area_data = compute_surface_area(vertices_surface, faces_surface)
    else:
        metric_area_data = nib.load(surface_path[1]).darrays[0].data

    metric_infile = nib.load(metric_path)
    metric_label = nib.load(str(output_prefix) + '.threshold.label.func.gii')
    metric_atlas = nib.load(atlas_path)

    metric_surface_data = metric_surface.darrays[0].data
    metric_atlas_data = metric_atlas.darrays[0].data
    metric_atlas_data_map = metric_atlas.labeltable.get_labels_as_dict()

    len_infile_label_equal = True
    if len(metric_infile.darrays) > len(metric_label.darrays):
        len_infile_label_equal = False
        print("will use label file multi time")
    elif len(metric_infile.darrays) < len(metric_label.darrays):
        ValueError("metric file time size is less than label file")

    for darray_idx in range(len(metric_infile.darrays)):
        cluster_all = []
        cluster_area_info_all = []
        metric_infile_c = metric_infile.darrays[darray_idx].data
        if len_infile_label_equal:
            metric_label_c = metric_label.darrays[darray_idx].data
        else:
            metric_label_c = metric_label.darrays[0].data
        for label_idx in range(np.int64(np.max(metric_label_c))):
            if label_idx + 1 not in np.unique(metric_label_c):
                continue

            if less_than:
                peak_vertex_idx = np.array(range(len(metric_label_c)))[metric_label_c == (label_idx + 1)][np.argmin(
                    metric_infile_c[metric_label_c == (label_idx + 1)])]

            else:
                peak_vertex_idx = np.array(range(len(metric_label_c)))[metric_label_c == (label_idx + 1)][np.argmax(
                    metric_infile_c[metric_label_c == (label_idx + 1)])]

            peak_vertex_value = metric_infile_c[peak_vertex_idx]
            peak_vertex_coord = metric_surface_data[peak_vertex_idx]
            peak_region = metric_atlas_data_map[metric_atlas_data[peak_vertex_idx]]
            cluster_area = np.sum(metric_area_data[metric_label_c == (label_idx + 1)])

            cluster_all.append(
                [label_idx+1, peak_vertex_coord[0], peak_vertex_coord[1], peak_vertex_coord[2],
                 peak_vertex_value, cluster_area, peak_region])

            for inter_label_idx in np.unique(metric_atlas_data[metric_label_c == (label_idx + 1)]):
                cluster_area_info_all.append([
                    label_idx + 1, metric_atlas_data_map[inter_label_idx],
                    np.sum(metric_area_data[
                               (metric_label_c == (label_idx + 1)) & (metric_atlas_data == inter_label_idx)
                           ]),
                    ])
        cluster_all = pd.DataFrame(
            cluster_all,
            columns=['cluster idx', 'coord-x', 'coord-y', 'coord-z', 'peak value', 'area', 'annot']
        )

        cluster_all.to_csv(output_prefix + f'_darray-{darray_idx}_cluster_info.csv', index=False)
        cluster_area_info_all = pd.DataFrame(
            cluster_area_info_all,
            columns=['cluster idx', 'region', 'area']
        )
        cluster_area_info_all.to_csv(output_prefix + f'_darray-{darray_idx}_cluster_info_area.csv', index=False)
