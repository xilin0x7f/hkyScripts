# Author: 赩林, xilin0x7f@163.com
import numpy as np
import nibabel as nib

def surface_get_parc_coord(surf_path, parc_path, out_path):
    import pandas as pd
    surf_header = nib.load(surf_path)
    parc_header = nib.load(parc_path)

    surf_vertices = surf_header.darrays[0].data
    # surf_faces = surf_header.darrays[1].data

    parc_data = parc_header.darrays[0].data
    parc_ids = [label[0] for label in parc_header.labeltable.get_labels_as_dict().items()]
    parc_names = [label[1] for label in parc_header.labeltable.get_labels_as_dict().items()]
    points_all = np.zeros((len(parc_ids), 3))
    for idx in range(len(parc_ids)):
        parc_cords = surf_vertices[parc_data == parc_ids[idx], :]
        if parc_cords.shape[0] == 0:
            points_all[idx, :] = [0, 0, 0]
            continue
        parc_cords_mean = np.mean(parc_cords, 0)
        distances = np.linalg.norm(parc_cords - parc_cords_mean, axis=1)
        points_all[idx, :] = parc_cords[np.argmin(distances)]

    df = pd.concat(
        [
            pd.DataFrame(parc_names, columns=['name']),
            pd.DataFrame(points_all, columns=['x', 'y', 'z'])
        ], axis=1
    )
    df.to_csv(out_path, index=False)

def compute_surface_area(vertices, faces):
    vectors = np.diff(vertices[faces], axis=1)
    cross = np.cross(vectors[:, 0], vectors[:, 1])
    areas = np.bincount(
        faces.flatten(),
        weights=np.repeat(np.sqrt(np.sum(cross ** 2, axis=1)) / 6, 3)
    )

    return areas
