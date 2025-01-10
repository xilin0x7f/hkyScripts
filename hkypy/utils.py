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

def get_matrix_tril(out_path, diag=False, matrix_paths=None):
    res = []
    for matrix_path in matrix_paths:
        matrix = np.loadtxt(matrix_path)
        res.append(matrix[np.triu_indices_from(matrix, k=0 if diag else 1)])

    np.savetxt(out_path, np.array(res))

def matrix_tril_to_matrix(tril_path, diag=False, out_prefix='matrix'):
    tril = np.loadtxt(tril_path)
    if tril.ndim == 1:
        tril = tril.reshape(1, -1)
    width = np.ceil(np.log10(tril.shape[0])).astype(int)
    width = width + 1 if width == 0 else width
    shape = np.round(0.5 + np.sqrt(1 + 8 * tril.shape[1])/2).astype(int)
    shape = shape - 1 if diag else shape
    for i in range(tril.shape[0]):
        matrix = np.zeros((shape, shape))
        matrix[np.triu_indices_from(matrix, k=0 if diag else 1)] = tril[i, :]
        matrix = matrix + matrix.T
        matrix = matrix - np.diag(np.diag(matrix))/2 if diag else matrix
        np.savetxt(f'{out_prefix}{i+1:0{width}d}.txt', matrix)
