import math
from math import sqrt
from typing import Tuple

import cv2
import dlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import feature_extraction.triangulation.conversions as tri_utils
from feature_extraction.faces.face_detector import FaceDetector
from feature_extraction.local_binary_pattern.local_binary_pattern import LocalBinaryPattern
from feature_extraction.utils.conversions import rect_to_bounding_box


def compute_triangle_area(triangle_points: np.ndarray) -> float:
    """ Compute area of a triangle given 3 points

    :param triangle_points: a numpy array of points
    :return: the area
    """
    (x1, y1), (x2, y2), (x3, y3) = tri_utils.unpack_triangle_coordinates(triangle_points)
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


def compute_mean_triangles_area(source_triangles_points: np.ndarray, dest_triangles_points: np.ndarray) -> np.ndarray:
    """ Compute the mean area between all triangles

    :param source_triangles_points: numpy source array of triangles
    :param dest_triangles_points: numpy dest array of triangles
    :return: the mean area rounded to second decimal
    """
    area_differences = []
    source_triangle_number = len(source_triangles_points)
    dest_triangle_number = len(dest_triangles_points)
    if source_triangle_number != dest_triangle_number:
        raise AssertionError('Images have different number of triangles')
    for t1, t2 in zip(source_triangles_points, dest_triangles_points):
        source_area = compute_triangle_area(t1)
        dest_area = compute_triangle_area(t2)
        area_differences.append(abs(source_area - dest_area))
    return np.array(area_differences)


def compute_triangle_centroid(triangle_points: np.ndarray) -> Tuple[float, float]:
    """ Compute centroid of a triangle given 3 points

    :param triangle_points: a numpy array of points
    :return: the centroid
    """
    (x1, y1), (x2, y2), (x3, y3) = tri_utils.unpack_triangle_coordinates(triangle_points)
    # Centroid
    centroid = (((x1 + x2 + x3) / 3), ((y1 + y2 + y3) / 3))
    return centroid


def compute_mean_centroids_distances(source_triangles_points: np.ndarray, dest_triangles_points: np.ndarray):
    """ Compute the mean distances between all centroids

        :param source_triangles_points: numpy source array of triangles
        :param dest_triangles_points: numpy dest array of triangles
        :return: the mean centroids distances rounded to second decimal
        """
    centroid_distances = []
    source_triangle_number = len(source_triangles_points)
    dest_triangle_number = len(dest_triangles_points)
    if source_triangle_number != dest_triangle_number:
        raise AssertionError('Images have different number of triangles')
    for t1, t2 in zip(source_triangles_points, dest_triangles_points):
        (c1_x, c1_y) = compute_triangle_centroid(t1)
        (c2_x, c2_y) = compute_triangle_centroid(t2)
        dist = sqrt((c2_x - c1_x) ** 2 + (c2_y - c1_y) ** 2)
        centroid_distances.append(dist)
    return np.array(centroid_distances)


def compute_triangle_angles(triangle_points: np.ndarray) -> np.ndarray:
    """ Compute angles of a triangle given 3 points

    :param triangle_points: a numpy array of points
    :return: the angles
    """
    (x1, y1), (x2, y2), (x3, y3) = tri_utils.unpack_triangle_coordinates(triangle_points)
    # Centroid
    dx_12, dy_12 = (x2 - x1), (y2 - y1)
    dx_23, dy_23 = (x3 - x2), (y3 - y2)
    dx_31, dy_31 = (x1 - x3), (y1 - y3)
    theta_12 = math.atan2(dy_12, dx_12)
    theta_23 = math.atan2(dy_23, dx_23)
    theta_31 = math.atan2(dy_31, dx_31)
    return np.array([theta_12, theta_23, theta_31])


def compute_mean_angles_distances(source_triangles_points: np.ndarray, dest_triangles_points: np.ndarray):
    """ Compute the mean angle distances between all triangles

        :param source_triangles_points: numpy source array of triangles
        :param dest_triangles_points: numpy dest array of triangles
        :return: the mean angles distances rounded to second decimal
        """
    cosine_distances = []
    source_triangle_number = len(source_triangles_points)
    dest_triangle_number = len(dest_triangles_points)
    if source_triangle_number != dest_triangle_number:
        raise AssertionError('Images have different number of triangles')
    for t1, t2 in zip(source_triangles_points, dest_triangles_points):
        tri_angles1 = compute_triangle_angles(t1)
        tri_angles2 = compute_triangle_angles(t2)
        similarity = cosine_similarity([tri_angles1], [tri_angles2])[0][0]
        cosine_distance = 1 - similarity
        cosine_distances.append(cosine_distance)
    return np.array(cosine_distances)


def compute_affine_matrix(source_triangle_points: np.ndarray, dest_triangle_points: np.ndarray) -> np.ndarray:
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


def compute_mean_affine_matrices_distances(source1_triangles_points: np.ndarray,
                                           source2_triangles_points: np.ndarray):
    """ Compute the mean affine matrices distances between source1, source2, target

    :param source1_triangles_points: numpy source1 array of triangles
    :param source2_triangles_points: numpy source2 array of triangles
    :param target_triangles_points: numpy target array of triangles
    :return: the mean affine matrices distances rounded to second decimal
    """
    matrices_distances = []
    source1_triangle_number = len(source1_triangles_points)
    source2_triangle_number = len(source2_triangles_points)
    if source1_triangle_number != source2_triangle_number:
        raise AssertionError('Images have different number of triangles')
    for t1, t2 in zip(source1_triangles_points, source2_triangles_points):
        affine_matrix_1 = compute_affine_matrix(t1, t2)
        matrices_distances.append(affine_matrix_1.flatten())
    return np.array(matrices_distances).flatten()


def compute_face_lbp_difference(source_img: np.ndarray, dest_img: np.ndarray, detector: FaceDetector,
                                lpb_extractor: LocalBinaryPattern):
    bbox_source: dlib.rectangle = detector.get_faces_bbox(source_img)[0]
    bbox_dest: dlib.rectangle = detector.get_faces_bbox(dest_img)[0]
    (x1, y1, w1, h1) = rect_to_bounding_box(bbox_source)
    (x2, y2, w2, h2) = rect_to_bounding_box(bbox_dest)
    source_img_crop = source_img.copy()[y1:y1 + h1, x1:x1 + w1]
    dest_img_crop = dest_img.copy()[y2:y2 + h2, x2:x2 + w2]
    source_img_crop = cv2.cvtColor(source_img_crop, cv2.COLOR_RGB2GRAY)
    dest_img_crop = cv2.cvtColor(dest_img_crop, cv2.COLOR_RGB2GRAY)
    # get_images_mosaic_no_labels("Mosaic", [source_img_crop, dest_img_crop], 1, 2).show()
    lbp_source = lpb_extractor.describe(source_img_crop)
    lbp_dest = lpb_extractor.describe(dest_img_crop)
    lbp_complete = np.concatenate([lbp_source.flatten(), lbp_dest.flatten()])
    return lbp_complete
