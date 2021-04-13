from math import sqrt

import numpy as np


def compute_triangle_area(triangle_points: np.ndarray) -> float:
    """ Compute area of a triangle given 3 points

    :param triangle_points: a numpy array of points
    :return: the area
    """
    x1, y1 = triangle_points[0], triangle_points[1]
    x2, y2 = triangle_points[2], triangle_points[3]
    x3, y3 = triangle_points[4], triangle_points[5]
    # Pythagorean theorem
    l1 = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    l2 = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    l3 = sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    # Heron's Formula
    semi_perimeter = (l1 + l2 + l3) / 2
    area = sqrt(semi_perimeter * (semi_perimeter - l1) * (semi_perimeter - l2) * (semi_perimeter - l3))
    return area


def compute_mean_triangles_area(triangles_points: np.ndarray):
    """ Compute the mean area between all triangles

    :param triangles_points: numpy array of triangles
    :return: the mean area rounded to second decimal
    """
    mean_area: float = 0
    triangle_number = len(triangles_points)
    for t in triangles_points:
        area = compute_triangle_area(t)
        mean_area += area
    mean_area = mean_area / triangle_number
    return round(mean_area, 2)
