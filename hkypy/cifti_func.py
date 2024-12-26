# Author: 赩林, xilin0x7f@163.com
import numpy as np
import nibabel as nib

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
    left_surface_path, right_surface_path, output_prefix, less_than=False, direction='COLUMN'
):
    raise NotImplementedError

def cifti_report():
    raise NotImplementedError
