import numpy as np

from descriptors.triangle_descriptors.triangle_descriptors import compute_triangle_affine_matrix


def compute_affine_matrices_descriptor(source1_triangles_points: np.ndarray,
                                       source2_triangles_points: np.ndarray):
    """ Compute all the affine matrices between all pairs of triangles of
        corresponding triangles in source and destination image Delaunay triangulation.

    **Descriptor**

    The descriptor is obtained concatenating all affine matrices flattened [m1, m2, m3, m4, ...]

    **Descriptor Idea**

    The idea is that affine matrices describe well the transformation between a source triangle and a destination triangle

    :param source1_triangles_points: numpy source1 array of triangles
    :param source2_triangles_points: numpy source2 array of triangles
    :return: the mean affine matrices distances rounded to second decimal
    """
    matrices_distances = []
    source1_triangle_number = len(source1_triangles_points)
    source2_triangle_number = len(source2_triangles_points)
    if source1_triangle_number != source2_triangle_number:
        raise AssertionError('Images have different number of triangles')
    for t1, t2 in zip(source1_triangles_points, source2_triangles_points):
        affine_matrix_1 = compute_triangle_affine_matrix(t1, t2)
        matrices_distances.append(affine_matrix_1.flatten())
    return np.array(matrices_distances).flatten()
