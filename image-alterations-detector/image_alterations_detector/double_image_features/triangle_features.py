import math
from math import sqrt
from typing import Tuple

import numpy as np

from face_morphology.landmarks_triangulation.conversions import unpack_triangle_coordinates


def compute_triangle_area(triangle_points: np.ndarray) -> float:
    """ Compute area of a triangle given 3 points

    :param triangle_points: a numpy array of points
    :return: the area
    """
    (x1, y1), (x2, y2), (x3, y3) = unpack_triangle_coordinates(triangle_points)
    # Pythagorean theorem
    l1 = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    l2 = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    l3 = sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    # Heron's Formula
    semi_perimeter = (l1 + l2 + l3) / 2
    to_sqrt = semi_perimeter * (semi_perimeter - l1) * (semi_perimeter - l2) * (semi_perimeter - l3)
    to_sqrt = to_sqrt if to_sqrt > 0 else 0
    area = sqrt(to_sqrt)
    return area


def compute_triangle_centroid(triangle_points: np.ndarray) -> Tuple[float, float]:
    """ Compute centroid of a triangle given 3 points

    :param triangle_points: a numpy array of points
    :return: the centroid
    """
    (x1, y1), (x2, y2), (x3, y3) = unpack_triangle_coordinates(triangle_points)
    # Centroid
    centroid = (((x1 + x2 + x3) / 3), ((y1 + y2 + y3) / 3))
    return centroid


def compute_triangle_angles(triangle_points: np.ndarray) -> np.ndarray:
    """ Compute angles of a triangle given 3 points

    :param triangle_points: a numpy array of points
    :return: the angles
    """
    (x1, y1), (x2, y2), (x3, y3) = unpack_triangle_coordinates(triangle_points)
    # Centroid
    dx_12, dy_12 = (x2 - x1), (y2 - y1)
    dx_23, dy_23 = (x3 - x2), (y3 - y2)
    dx_31, dy_31 = (x1 - x3), (y1 - y3)
    theta_12 = math.atan2(dy_12, dx_12)
    theta_23 = math.atan2(dy_23, dx_23)
    theta_31 = math.atan2(dy_31, dx_31)
    return np.array([theta_12, theta_23, theta_31])