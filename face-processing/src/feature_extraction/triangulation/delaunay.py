from typing import List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

import feature_extraction.triangulation.utils as utils


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


if __name__ == '__main__':
    from feature_extraction.landmarks.landmark_extractor import LandmarkExtractor

    img = cv2.imread('../../../images/img1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    extractor = LandmarkExtractor("../../../models/shape_predictor_68_face_landmarks.dat")
    points = extractor.get_2d_landmarks(img)
    triangles_indexes = get_triangulations_indexes(img, points)
    image_delaunay = utils.draw_delaunay(img, points, triangles_indexes, (150, 0, 0))
    plt.imshow(image_delaunay)
    plt.show()
