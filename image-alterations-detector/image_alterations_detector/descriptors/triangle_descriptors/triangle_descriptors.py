import math
from math import sqrt

import cv2
import numpy as np
from numpy.linalg import norm

from image_alterations_detector.face_morphology.landmarks_triangulation.conversions import unpack_triangle_coordinates


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


def compute_triangle_side_centroid_distances(triangle_points: np.ndarray) -> np.ndarray:
    """ Compute centroid of a triangle given 3 points

    :param triangle_points: a numpy array of points
    :return: the centroid
    """

    def distance_point_line(point, p1, p2):
        norm_p2_p1 = norm(p2 - p1)
        norm_p2_p1 = norm_p2_p1 if norm(p2 - p1) != 0 else np.finfo(float).eps
        d = np.abs(norm(np.cross(p2 - p1, p1 - point))) / norm_p2_p1
        return d

    (x1, y1), (x2, y2), (x3, y3) = unpack_triangle_coordinates(triangle_points)
    # Centroid
    centroid = (((x1 + x2 + x3) / 3), ((y1 + y2 + y3) / 3))
    relative_centroid_1_2 = distance_point_line(np.array(centroid), np.array([x1, y1]), np.array([x2, y2]))
    relative_centroid_2_3 = distance_point_line(np.array(centroid), np.array([x2, y2]), np.array([x3, y3]))
    relative_centroid_3_1 = distance_point_line(np.array(centroid), np.array([x3, y3]), np.array([x1, y1]))
    centroid_distances_to_sides = np.array([relative_centroid_1_2, relative_centroid_2_3, relative_centroid_3_1])
    return centroid_distances_to_sides


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


def compute_triangle_affine_matrix(source_triangle_points: np.ndarray, dest_triangle_points: np.ndarray) -> np.ndarray:
    """ Compute the mean of all cosine distance between all pairs of angles (a1, b1, c1) (a2, b2, c2) of
           corresponding triangles in source and destination image Delaunay triangulation

    :param source_triangle_points: numpy source array of triangles
    :param dest_triangle_points: numpy dest array of triangles
    :return: the mean angles distances rounded to second decimal
    """
    pt1_source = source_triangle_points[0], source_triangle_points[1]
    pt2_source = source_triangle_points[2], source_triangle_points[3]
    pt3_source = source_triangle_points[4], source_triangle_points[5]
    pt1_dest = dest_triangle_points[0], dest_triangle_points[1]
    pt2_dest = dest_triangle_points[2], dest_triangle_points[3]
    pt3_dest = dest_triangle_points[4], dest_triangle_points[5]

    pts_source = np.float32([pt1_source, pt2_source, pt3_source])
    pts_dest = np.float32([pt1_dest, pt2_dest, pt3_dest])
    warping_matrix = cv2.getAffineTransform(pts_source, pts_dest)
    return warping_matrix


if __name__ == '__main__':
    dist = compute_triangle_side_centroid_distances(np.array([0, -4 / 3, 2, 0, 5, 6]))
    print(dist)
