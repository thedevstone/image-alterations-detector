# Check if a point is inside a rectangle
from typing import Tuple, List

import cv2
import numpy as np


def points_to_list_of_tuple(points: np.ndarray) -> List[Tuple[int, int]]:
    points_list = []
    for i, p in enumerate(points):
        x = points[i][0]
        y = points[i][1]
        points_list.append((x, y))
    return points_list


def triangulation_indexes_to_points(points: np.ndarray, triangles_indexes: np.ndarray) -> np.ndarray:
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


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay_from_indexes(img: np.ndarray, points: np.ndarray, triangles_indexes: np.ndarray,
                               delaunay_color: Tuple[int, int, int]) -> np.ndarray:
    # Get shape of image
    size = img.shape
    r = (0, 0, size[1], size[0])
    # List of points
    triangles = triangulation_indexes_to_points(points, triangles_indexes)
    # Output image
    image_out = img.copy()
    # Find points coordinates
    for t in triangles:
        tri_point1 = (t[0], t[1])
        tri_point2 = (t[2], t[3])
        tri_point3 = (t[4], t[5])

        if rect_contains(r, tri_point1) and rect_contains(r, tri_point2) and rect_contains(r, tri_point3):
            cv2.line(image_out, tri_point1, tri_point2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(image_out, tri_point2, tri_point3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(image_out, tri_point3, tri_point1, delaunay_color, 1, cv2.LINE_AA, 0)
    return image_out
