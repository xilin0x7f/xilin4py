# Author: 赩林, xilin0x7f@163.com
import nibabel as nb
import numpy as np

def compute_surface_area(vertices, faces):
    vertex_areas = np.zeros(vertices.shape[0])
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        vertex_areas[face] += np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 6.0

    return vertex_areas

def compute_surface_area2(vertices, faces):
    vectors = np.diff(vertices[faces], axis=1)
    cross = np.cross(vectors[:, 0], vectors[:, 1])
    areas = np.bincount(
        faces.flatten(),
        weights=np.repeat(np.sqrt(np.sum(cross ** 2, axis=1)) / 6, 3)
    )

    return areas

gifti_file = nb.load(r'C:\AppsData\OneDrive\data\templates\fsLR32k\tpl-fsLR_den-32k_hemi-L_inflated.surf.gii')
vertices, faces = gifti_file.darrays[0].data, gifti_file.darrays[1].data