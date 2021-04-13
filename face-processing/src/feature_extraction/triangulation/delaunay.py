from typing import List

import cv2
import numpy as np


def get_triangulations(img: np.ndarray, points: np.ndarray) -> List[np.ndarray]:
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
    triangle_list: List[np.ndarray] = subdivision.getTriangleList()
    return triangle_list


def get_triangulations_indexes(img: np.ndarray, points: np.ndarray):
    """ Get triangulation indexes bases on points list

    :param img: the input image
    :param points: the landmark points
    :return: a list of tuples ot triangles points
    """
    triangle_list = get_triangulations(img, points)
    points_list = []
    triangle_indexes = []
    for i, p in enumerate(points):
        x = points[i][0]
        y = points[i][1]
        points_list.append((x, y))

    for t in triangle_list:
        # Get triangles points
        tri_point1 = (t[0], t[1])
        tri_point2 = (t[2], t[3])
        tri_point3 = (t[4], t[5])
        # Get index of points
        index_tri_pt1 = points_list.index(tri_point1)
        index_tri_pt2 = points_list.index(tri_point2)
        index_tri_pt3 = points_list.index(tri_point3)
        # Append indexes
        triangle_indexes.append((index_tri_pt1, index_tri_pt2, index_tri_pt3))
    return triangle_indexes


if __name__ == '__main__':
    from feature_extraction.landmarks.landmark_extractor import LandmarkExtractor

    img = cv2.imread('../../../images/img1.jpg')
    extractor = LandmarkExtractor("../../../models/shape_predictor_68_face_landmarks.dat")
    points = extractor.get_2d_landmarks(img)
    get_triangulations_indexes(img, points)
