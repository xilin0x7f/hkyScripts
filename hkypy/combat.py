# Author: 赩林, xilin0x7f@163.com
import pandas as pd
import numpy as np

def combat(data4combat, covars, batch='site', con=None, cat=None, eb=True, factor=0):
    from neuroCombat import neuroCombat
    if factor == 0:
        if len(data4combat[data4combat > 0]) == 0 and len(data4combat[data4combat < 0]) == 0:
            raise ValueError('data4combat is all zero')
        else:
            median_pos, median_neg = np.nan, np.nan
            if len(data4combat[data4combat > 0]) != 0:
                median_pos = np.median(data4combat[data4combat > 0])
            if len(data4combat[data4combat < 0]) != 0:
                median_neg = np.median(data4combat[data4combat < 0])

        factor = round(
            1000 * (2 - np.isnan([median_pos, median_neg]).sum()) /
            np.nansum([median_pos, -1 * median_neg])
        )

    data_harm = neuroCombat(
        dat=data4combat.T * factor, covars=covars, batch_col=batch, categorical_cols=cat, continuous_cols=con,
        eb=eb
    )["data"].T / factor

    return data_harm

def text_combat(
    text_path, subj_info_path, out_path, batch='site', con=None, cat=None,
    eb=True, factor=0
):
    data4combat = np.loadtxt(text_path)
    if data4combat.ndim == 1:
        data4combat = data4combat.reshape(-1, 1)
        eb = False

    subj_info = pd.read_csv(subj_info_path)

    if con is None and cat is None:
        covars = subj_info
    else:
        col = [batch]
        if con is not None:
            con = con.split(',')
            col = col + con
        if cat is not None:
            cat = cat.split(',')
            col = col + cat

        covars = subj_info[col]

    data_harm = combat(data4combat, covars, batch, con, cat, eb, factor)
    np.savetxt(out_path, data_harm)

def volume_combat(
    volume_path, mask_path, subj_info_path, out_path, batch='site', con=None, cat=None, eb=True, factor=0
):
    from .volume_func import volume_extract, volume_restore

    data4combat = volume_extract(volume_path, mask_path)
    subj_info = pd.read_csv(subj_info_path)

    if con is None and cat is None:
        covars = subj_info
    else:
        col = [batch]
        if con is not None:
            con = con.split(',')
            col = col + con
        if cat is not None:
            cat = cat.split(',')
            col = col + cat

        covars = subj_info[col]

    data_harm = combat(data4combat.T, covars, batch, con, cat, eb, factor)
    volume_restore(data_harm.T, mask_path, out_path)
