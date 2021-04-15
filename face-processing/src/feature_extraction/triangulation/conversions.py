# Check if a point is inside a rectangle
from typing import Tuple, List

import numpy as np


def unpack_triangle_coordinates(triangle_points: np.ndarray) -> \
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """ Unpack a 6-elem array into 3 points of a triangle

    :param triangle_points: a 6-elem numpy array with (x, y) of 3 points of a triangle
    :return: a tuple of three points of the triangle
    """
    x1, y1 = triangle_points[0], triangle_points[1]
    x2, y2 = triangle_points[2], triangle_points[3]
    x3, y3 = triangle_points[4], triangle_points[5]
    p1 = (x1, y1)
    p2 = (x2, y2)
    p3 = (x3, y3)
    return p1, p2, p3


def points_to_list_of_tuple(points: np.ndarray) -> List[Tuple[int, int]]:
    """ Convert a numpy array to a list of tuple 2D

    :param points: the numpy array of 2D points
    :return: the list of tuple 2D
    """
    points_list = []
    for i, p in enumerate(points):
        x = points[i][0]
        y = points[i][1]
        points_list.append((x, y))
    return points_list


def triangulation_indexes_to_points(points: np.ndarray, triangles_indexes: np.ndarray) -> np.ndarray:
    """ Convert triangles indexes to real points given a numpy array of actual points

    :param points: the numpy array of points
    :param triangles_indexes: the numpy array of indexes
    :return: a numpy array of the triangles formed by points present in points array
    """
    # List of points
    points_list = points_to_list_of_tuple(points)
    triangles = np.zeros((len(triangles_indexes), 6), dtype='int')
    for i, t in enumerate(triangles_indexes):
        tri_point1: Tuple[int, int] = points_list[t[0]]
        tri_point2: Tuple[int, int] = points_list[t[1]]
        tri_point3: Tuple[int, int] = points_list[t[2]]
        triangles[i] = [tri_point1[0],
                        tri_point1[1],
                        tri_point2[0],
                        tri_point2[1],
                        tri_point3[0],
                        tri_point3[1]]
    return triangles
