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

def dmri_dki_fit(dwi_path, bvec_path, bval_path, mask_path, out_prefix="dki_", fwhm=1.25):
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.reconst.dki import DiffusionKurtosisModel
    from scipy.ndimage import gaussian_filter

    img = nib.load(dwi_path)
    data = img.get_fdata()
    affine = img.affine
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata().astype(bool)
    gtab = gradient_table(*read_bvals_bvecs(bval_path, bvec_path))

    if fwhm > 0:
        data_smooth = np.zeros(data.shape)
        for v in range(data.shape[-1]):
            data_smooth[..., v] = gaussian_filter(data[..., v], sigma=fwhm/np.sqrt(8 * np.log(2)))
    else:
        data_smooth = data

    dki_model = DiffusionKurtosisModel(gtab)
    dki_fit = dki_model.fit(data_smooth, mask=mask)

    nib.save(nib.Nifti1Image(dki_fit.fa, affine), out_prefix+"_FA.nii.gz")
    nib.save(nib.Nifti1Image(dki_fit.md, affine), out_prefix+"_MD.nii.gz")
    nib.save(nib.Nifti1Image(dki_fit.ad, affine), out_prefix+"_AD.nii.gz")
    nib.save(nib.Nifti1Image(dki_fit.rd, affine), out_prefix+"_RD.nii.gz")
    nib.save(nib.Nifti1Image(dki_fit.mk, affine), out_prefix+"_MK.nii.gz")
    nib.save(nib.Nifti1Image(dki_fit.ak, affine), out_prefix+"_AK.nii.gz")
    nib.save(nib.Nifti1Image(dki_fit.rk, affine), out_prefix+"_RK.nii.gz")
