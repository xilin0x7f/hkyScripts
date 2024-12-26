# Author: 赩林, xilin0x7f@163.com
import numpy as np
import pandas as pd

def get_motion(output_path, files_path):
    motion_all = []
    for file_idx, file_path in enumerate(files_path):
        motions = pd.read_csv(file_path, delimiter='\t')
        mean_fd = motions['framewise_displacement'].mean()
        max_rot = max([motions[f'rot_{direction}'].abs().max() for direction in ('x', 'y', 'z')]) * 180.0 / np.pi
        max_trans = max([motions[f'trans_{direction}'].abs().max() for direction in ('x', 'y', 'z')])
        motion_all.append([file_path, mean_fd, max_rot, max_trans])

    motion_all = pd.DataFrame(motion_all,  columns=['file_path', 'mean_fd', 'max_rot', 'max_trans'])

    motion_all.to_csv(output_path, index=False)
