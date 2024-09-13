# Author: 赩林, xilin0x7f@163.com
import numpy as np
import nibabel as nib

def compute_surface_area(vertices, faces):
    vectors = np.diff(vertices[faces], axis=1)
    cross = np.cross(vectors[:, 0], vectors[:, 1])
    areas = np.bincount(
        faces.flatten(),
        weights=np.repeat(np.sqrt(np.sum(cross ** 2, axis=1)) / 6, 3)
    )

    return areas

def cifti_separate(cifti_file):
    cifti_data = cifti_file.get_fdata()
    brain_models = cifti_file.header.matrix.get_index_map(1).brain_models

    volume_data, volume_mat = None, None
    ax1 = cifti_file.header.get_axis(1)
    if ax1._volume_shape is not None:
        volume_dim = ax1._volume_shape
        volume_mat = ax1._affine
        volume_data = np.zeros(list(volume_dim) + [cifti_data.shape[0]])

    cortex_left_data, cortex_left_indices = None, None
    cortex_right_data, cortex_right_indices = None, None

    for bm in brain_models:
        if bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
            cortex_left_data = np.zeros([cifti_data.shape[0], bm.surface_number_of_vertices])
            cortex_left_indices = bm._vertex_indices._indices
            cortex_left_data[:, cortex_left_indices] = cifti_data[:, bm.index_offset:bm.index_offset + bm.index_count]

        elif bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            cortex_right_data = np.zeros([cifti_data.shape[0], bm.surface_number_of_vertices])
            cortex_right_indices = bm._vertex_indices._indices
            cortex_right_data[:, cortex_right_indices] = cifti_data[:, bm.index_offset:bm.index_offset + bm.index_count]

        elif bm.model_type == "CIFTI_MODEL_TYPE_VOXELS":
            indices = np.array(bm._voxel_indices_ijk._indices)
            volume_data[indices[:, 0], indices[:, 1], indices[:, 2], :] = cifti_data[:, bm.index_offset:bm.index_offset + bm.index_count].T

    return cortex_left_data, cortex_right_data, cortex_left_indices, cortex_right_indices, volume_data, volume_mat

def cifti_surface_zscore(cifti_file, mask_left=None, mask_right=None, weight_lh=None, weight_rh=None):
    cortex_left_data, cortex_right_data, cortex_left_indices, cortex_right_indices = cifti_separate(cifti_file)[:4]
    if mask_left is None:
        mask_left = np.zeros(cortex_left_data.shape[1])
        mask_left[cortex_left_indices] = 1

    if mask_right is None:
        mask_right = np.zeros(cortex_right_data.shape[1])
        mask_right[cortex_right_indices] = 1

    mask_left, mask_right = mask_left.flatten(), mask_right.flatten()
    cortex_all = np.hstack([cortex_left_data, cortex_right_data])
    mask_all = np.hstack([mask_left, mask_right])
    cortex_masked = cortex_all[:, mask_all > 0]
    if weight_lh is not None and weight_rh is not None:
        weight_all = np.hstack([weight_lh, weight_rh])
        weight_masked = weight_all[:, mask_all > 0]
        mean_value = np.average(cortex_masked, weights=weight_masked, axis=1)[:, np.newaxis]
        std_value = np.sqrt(np.average((cortex_masked-mean_value)**2, weights=weight_masked, axis=1))[:, np.newaxis]
        cortex_masked_zscore = (cortex_masked - mean_value) / std_value
    else:
        cortex_masked_zscore = (cortex_masked - np.nanmean(cortex_masked, axis=1)[:, np.newaxis]) / np.nanstd(cortex_masked, axis=1)[:, np.newaxis]

    # cortex_all_zscore = cortex_all * 0
    # cortex_all_zscore[:, mask_all > 0] = cortex_masked_zscore

    mask_left_vertex = np.array(range(cortex_left_data.shape[1]))
    mask_left_vertex = mask_left_vertex[mask_left > 0]

    mask_right_vertex = np.array(range(cortex_right_data.shape[1]))
    mask_right_vertex = mask_right_vertex[mask_right > 0]

    # ax0 = cifti_file.header.get_axis(0)
    ax0 = nib.cifti2.cifti2_axes.ScalarAxis(name=[f"#{i+1}" for i in range(cortex_all.shape[0])])
    # ax1 = cifti_file.header.get_axis(1)
    ax1 = nib.cifti2.cifti2_axes.BrainModelAxis(
        name=np.array(['CIFTI_STRUCTURE_CORTEX_LEFT' for i in range(np.sum(mask_left > 0))] +
                      ['CIFTI_STRUCTURE_CORTEX_RIGHT' for i in range(np.sum(mask_right > 0))]),
        nvertices={
            "CIFTI_STRUCTURE_CORTEX_LEFT": cortex_left_data.shape[1],
            "CIFTI_STRUCTURE_CORTEX_RIGHT": cortex_right_data.shape[1]},
        vertex=np.hstack([mask_left_vertex, mask_right_vertex])
    )
    header = nib.Cifti2Header.from_axes((ax0, ax1))
    img = nib.Cifti2Image(cortex_masked_zscore, header)
    img.nifti_header.set_intent('ConnDenseScalar')

    return img
