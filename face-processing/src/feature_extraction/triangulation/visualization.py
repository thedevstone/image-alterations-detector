from typing import Tuple

import cv2
import numpy as np


def rect_contains(rect, point):
    """ Check if a point is inside a given rectangle

    :param rect: the rectangle
    :param point: the point
    :return: True if the point is inside the rectangle otherwise returns False
    """
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay_from_triangles_points(img: np.ndarray, triangles_points: np.ndarray,
                                        delaunay_color: Tuple[int, int, int]) -> np.ndarray:
    """ Draw the Delaunay graph triangulation points

    :param img: the image
    :param triangles_points: the indexes of triangles
    :param delaunay_color: the drawing color
    :return: a new image with a drawn triangulation
    """
    # Get shape of image
    size = img.shape
    r = (0, 0, size[1], size[0])
    # Output image
    image_out = img.copy()
    # Find points coordinates
    for t in triangles_points:
        tri_point1 = (t[0], t[1])
        tri_point2 = (t[2], t[3])
        tri_point3 = (t[4], t[5])

        if rect_contains(r, tri_point1) and rect_contains(r, tri_point2) and rect_contains(r, tri_point3):
            cv2.line(image_out, tri_point1, tri_point2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(image_out, tri_point2, tri_point3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(image_out, tri_point3, tri_point1, delaunay_color, 1, cv2.LINE_AA, 0)
    return image_out
