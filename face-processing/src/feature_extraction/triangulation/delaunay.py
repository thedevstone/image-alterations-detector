from typing import List, Tuple

import cv2
import numpy as np

import feature_extraction.triangulation.utils as utils
import measures.triangles_measures as measures
from feature_extraction.landmarks.utils import get_indexes_group_from_key
from plotting.plotting import get_images_mosaic


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

    # Load images
    img1 = cv2.imread('../../../images/img1.jpg')
    img2 = cv2.imread('../../../images/img2.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # Extract landmark indexes
    extractor = LandmarkExtractor("../../../models/shape_predictor_68_face_landmarks.dat")
    points1 = extractor.get_2d_landmarks(img1)
    points2 = extractor.get_2d_landmarks(img2)
    # Extract indexes from one of the two
    triangles_indexes = get_triangulations_indexes(img1, points1)
    # Subset of features
    right_eye_indexes = get_indexes_group_from_key('right_eye')
    left_eye_indexes = get_indexes_group_from_key('left_eye')
    nose_indexes = get_indexes_group_from_key('nose')
    triangles_indexes_right_eye = get_triangulations_indexes_subset(triangles_indexes, right_eye_indexes)
    triangles_indexes_left_eye = get_triangulations_indexes_subset(triangles_indexes, left_eye_indexes)
    triangles_indexes_nose = get_triangulations_indexes_subset(triangles_indexes, nose_indexes)
    indexes_group_union = np.row_stack([triangles_indexes_right_eye,
                                        triangles_indexes_left_eye,
                                        triangles_indexes_nose])
    # Draw Delaunay
    triangles_group_union_points1 = utils.triangulation_indexes_to_points(points1, indexes_group_union)
    triangles_group_union_points2 = utils.triangulation_indexes_to_points(points2, indexes_group_union)
    image_delaunay1 = utils.draw_delaunay_from_triangles(img1, triangles_group_union_points1, (150, 0, 0))
    image_delaunay2 = utils.draw_delaunay_from_triangles(img2, triangles_group_union_points2, (150, 0, 0))
    images = [(image_delaunay1, 'Delaunay 1'), (image_delaunay2, 'Delaunay 2')]
    mosaic = get_images_mosaic('Delaunay', images, 1, 2)
    mosaic.show()

    # Compute area
    triangles_points1 = utils.triangulation_indexes_to_points(points1, triangles_indexes)
    triangles_points2 = utils.triangulation_indexes_to_points(points2, triangles_indexes)
    mean_area1 = measures.compute_mean_triangles_area(triangles_points1)
    mean_area2 = measures.compute_mean_triangles_area(triangles_points2)
    print(mean_area1, mean_area2)
