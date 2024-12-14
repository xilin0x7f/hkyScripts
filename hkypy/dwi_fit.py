# Author: 赩林, xilin0x7f@163.com

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
