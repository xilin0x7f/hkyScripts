# Author: 赩林, xilin0x7f@163.com
import numpy as np
import nibabel as nib

def dmri_amico_fit(dwi_path, bvec_path, bval_path, mask_path, model_name="NODDI"):
    import amico
    import os
    amico.core.setup()
    amico.util.fsl2scheme(bval_path, bvec_path)
    ae = amico.Evaluation()
    ae.load_data(
        dwi_filename=dwi_path, scheme_filename=os.path.splitext(bval_path)[0]+".scheme", mask_filename=mask_path,
        b0_thr=10
    )
    ae.set_model(model_name)
    ae.generate_kernels(regenerate=True)
    ae.load_kernels()
    ae.fit()
    ae.save_results()

def dmri_dki_fit(
    dwi_path, bvec_path, bval_path, mask_path, out_prefix="dki_", fwhm=1.25, method='WLS', min_kurtosis=0, max_kurtosis=3
):
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.reconst.dki import DiffusionKurtosisModel
    from scipy.ndimage import gaussian_filter

    img = nib.load(dwi_path)
    data = img.get_fdata()
    affine = img.affine
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata().astype(bool)
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs=bvecs)

    if fwhm > 0:
        data_smooth = np.zeros(data.shape)
        for v in range(data.shape[-1]):
            data_smooth[mask, v] = gaussian_filter(data[mask, v], sigma=fwhm/np.sqrt(8 * np.log(2)))
    else:
        data_smooth = data

    dki_model = DiffusionKurtosisModel(gtab, fit_method=method)
    dki_fit = dki_model.fit(data_smooth, mask=mask)
    for index in ['fa', 'md', 'ad', 'rd', 'kfa', 'mkt', 'mk', 'ak', 'rk']:
        index_data = getattr(dki_fit, index)
        if index in ['mkt', 'mk', 'ak', 'rk']:
            index_data = index_data(min_kurtosis=min_kurtosis, max_kurtosis=max_kurtosis)
        # index_data = np.array([
        #     [
        #         [0.0 if callable(x) or x is None else x for x in arr1] for arr1 in arr2
        #     ]
        #     for arr2 in index_data
        # ])
        nib.save(nib.Nifti1Image(index_data, affine), out_prefix + index.upper() + ".nii.gz")
