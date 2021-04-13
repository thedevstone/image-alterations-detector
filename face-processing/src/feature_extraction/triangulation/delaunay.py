from typing import List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

import feature_extraction.triangulation.utils as utils
from feature_extraction.landmarks.utils import get_indexes_group_from_key


def get_triangulations(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """ Get triangulation array
    
    :param img: the input image
    :param points: the keypoints
    :return: the triangulation array
    """
    # Get the subdivision area
    size = img.shape
    subdivision_area = (0, 0, size[1], size[0])
    # Get subdivision object
    subdivision: cv2.Subdiv2D = cv2.Subdiv2D(subdivision_area)
    # Inserting point into the subdivision object
    for p in points:
        p = tuple(p)
        subdivision.insert(p)
    # Get triangulation
    triangles: np.ndarray = subdivision.getTriangleList()
    return triangles


def get_triangulations_indexes(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """ Get triangulation indexes bases on points list

    :param img: the input image
    :param points: the landmark points
    :return: a list of tuples ot triangles points
    """
    triangles = get_triangulations(img, points).astype('int')
    triangles_indexes = np.zeros((len(triangles), 3), dtype='int')
    points_list: List[Tuple[int, int]] = utils.points_to_list_of_tuple(points)
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


def get_triangulations_indexes_subset(triangles_indexes: np.ndarray, indexes_group: np.ndarray):
    sub_triangles_indexes = []
    for t in triangles_indexes:
        if set(t).issubset(set(indexes_group)):
            sub_triangles_indexes.append(t)
    sub_triangles_indexes = np.array(sub_triangles_indexes)
    return sub_triangles_indexes


if __name__ == '__main__':
    from feature_extraction.landmarks.landmark_extractor import LandmarkExtractor

    img = cv2.imread('../../../images/img1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    extractor = LandmarkExtractor("../../../models/shape_predictor_68_face_landmarks.dat")
    points = extractor.get_2d_landmarks(img)
    triangles_indexes = get_triangulations_indexes(img, points)

    right_eye_indexes = get_indexes_group_from_key('right_eye')
    left_eye_indexes = get_indexes_group_from_key('left_eye')
    nose_indexes = get_indexes_group_from_key('nose')

    triangles_indexes_right_eye = get_triangulations_indexes_subset(triangles_indexes, right_eye_indexes)
    triangles_indexes_left_eye = get_triangulations_indexes_subset(triangles_indexes, left_eye_indexes)
    triangles_indexes_nose = get_triangulations_indexes_subset(triangles_indexes, nose_indexes)

    all_indexes = np.row_stack([triangles_indexes_right_eye, triangles_indexes_left_eye, triangles_indexes_nose])
    image_delaunay = utils.draw_delaunay_from_indexes(img, points, all_indexes, (150, 0, 0))
    plt.imshow(image_delaunay)
    plt.show()
