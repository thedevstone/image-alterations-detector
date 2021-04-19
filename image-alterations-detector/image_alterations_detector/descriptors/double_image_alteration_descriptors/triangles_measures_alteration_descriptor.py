from math import sqrt

import numpy as np

from image_alterations_detector.descriptors.triangle_descriptors.triangle_descriptors import compute_triangle_area, \
    compute_triangle_centroid, compute_triangle_angles


def compute_mean_triangles_area_differences_descriptor(source_triangles_points: np.ndarray,
                                                       dest_triangles_points: np.ndarray) -> np.ndarray:
    """ Compute the mean of all absolute differences between all pairs of areas (a1, a2) of
        corresponding triangles in source and destination image Delaunay triangulation

    :param source_triangles_points: numpy source array of triangles
    :param dest_triangles_points: numpy dest array of triangles
    :return: the mean area rounded to second decimal
    """
    area_differences = []
    source_triangle_number = len(source_triangles_points)
    dest_triangle_number = len(dest_triangles_points)
    if source_triangle_number != dest_triangle_number:
        raise AssertionError('Images have different number of triangles')
    for t1, t2 in zip(source_triangles_points, dest_triangles_points):
        source_area = compute_triangle_area(t1)
        dest_area = compute_triangle_area(t2)
        area_differences.append(abs(source_area - dest_area))
    return np.array(area_differences)


def compute_mean_triangles_centroids_distances_descriptor(source_triangles_points: np.ndarray,
                                                          dest_triangles_points: np.ndarray):
    """ Compute the mean of all distances between all pairs of centroids ((c1_x, c1_y), (c2_x, c2_y)) of
        corresponding triangles in source and destination image Delaunay triangulation

    :param source_triangles_points: numpy source array of triangles
    :param dest_triangles_points: numpy dest array of triangles
    :return: the mean centroids distances rounded to second decimal
    """
    centroid_distances = []
    source_triangle_number = len(source_triangles_points)
    dest_triangle_number = len(dest_triangles_points)
    if source_triangle_number != dest_triangle_number:
        raise AssertionError('Images have different number of triangles')
    for t1, t2 in zip(source_triangles_points, dest_triangles_points):
        (c1_x, c1_y) = compute_triangle_centroid(t1)
        (c2_x, c2_y) = compute_triangle_centroid(t2)
        dist = sqrt((c2_x - c1_x) ** 2 + (c2_y - c1_y) ** 2)
        centroid_distances.append(dist)
    return np.array(centroid_distances)


def compute_mean_triangles_angles_distances_descriptor(source_triangles_points: np.ndarray,
                                                       dest_triangles_points: np.ndarray):
    """ Compute the mean of all cosine distance between all pairs of angles (a1, b1, c1) (a2, b2, c2) of
        corresponding triangles in source and destination image Delaunay triangulation

    :param source_triangles_points: numpy source array of triangles
    :param dest_triangles_points: numpy dest array of triangles
    :return: the mean angles distances rounded to second decimal
    """
    angle_differences = []
    source_triangle_number = len(source_triangles_points)
    dest_triangle_number = len(dest_triangles_points)
    if source_triangle_number != dest_triangle_number:
        raise AssertionError('Images have different number of triangles')
    for t1, t2 in zip(source_triangles_points, dest_triangles_points):
        tri_angles1 = compute_triangle_angles(t1)
        tri_angles2 = compute_triangle_angles(t2)
        diff = np.abs(tri_angles1 - tri_angles2)
        angle_differences.append(diff)
    return np.array(angle_differences).flatten()
