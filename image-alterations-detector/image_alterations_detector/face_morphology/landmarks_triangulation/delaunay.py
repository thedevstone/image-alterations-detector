from typing import List, Tuple

import cv2
import numpy as np

from face_morphology.landmarks_triangulation.conversions import points_to_list_of_tuple


def compute_triangulation_from_landmarks(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """ Compute the Delaunay triangulations on the image according to given landmarks_prediction.
    
    :param img: the input image
    :param landmarks: the landmarks_prediction
    :return: the landmarks_triangulation array
    """
    # Get the subdivision area
    size = img.shape
    subdivision_area = (0, 0, size[1], size[0])
    # Get subdivision object
    subdivision: cv2.Subdiv2D = cv2.Subdiv2D(subdivision_area)
    # Inserting point into the subdivision object
    for p in landmarks:
        p = tuple(p)
        subdivision.insert(p)
    # Get landmarks_triangulation
    triangles_points: np.ndarray = subdivision.getTriangleList()
    return triangles_points


def compute_triangulation_indexes(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """ Get landmarks_triangulation indexes bases on points list

    :param img: the input image
    :param landmarks: the landmarks_prediction
    :return: a list of tuples ot triangles points
    """
    triangles = compute_triangulation_from_landmarks(img, landmarks)
    triangles_indexes = np.zeros((len(triangles), 3), dtype='int')
    points_list: List[Tuple[int, int]] = points_to_list_of_tuple(landmarks)
    for i, t in enumerate(triangles):
        # Get triangles points
        tri_point1 = (t[0], t[1])
        tri_point2 = (t[2], t[3])
        tri_point3 = (t[4], t[5])
        # Get index of points
        index_tri_pt1 = points_list.index(tri_point1)
        index_tri_pt2 = points_list.index(tri_point2)
        index_tri_pt3 = points_list.index(tri_point3)
        # Append indexes
        triangles_indexes[i] = [index_tri_pt1, index_tri_pt2, index_tri_pt3]
    return triangles_indexes


def get_triangulations_indexes_subset(triangles_indexes: np.ndarray, indexes_group: np.ndarray) -> np.ndarray:
    """ Return landmarks_triangulation indexes from a subset of points

    :param triangles_indexes: the landmarks_triangulation indexes
    :param indexes_group: the subset of points. For example eyes area
    :return: the numpy array of indexes
    """
    sub_triangles_indexes = []
    for t in triangles_indexes:
        if set(t).issubset(set(indexes_group)):
            sub_triangles_indexes.append(t)
    sub_triangles_indexes = np.array(sub_triangles_indexes)
    return sub_triangles_indexes
