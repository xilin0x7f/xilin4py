# Author: 赩林, xilin0x7f@163.com
import numpy as np

def compute_surface_area(vertices, faces):
    vertex_areas = np.zeros(vertices.shape[0])
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        vertex_areas[face] = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 6.0

    return vertex_areas
