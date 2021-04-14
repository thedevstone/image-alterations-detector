from typing import List, Tuple

import cv2
import numpy as np

import face_align.face_aligner as aligner
import feature_extraction.triangulation.utils as utils
import measures.triangles_measures as measures
from plotting.plotting import get_images_mosaic_with_label


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
    img1 = cv2.imread('../../../images/m-004-1.png')
    img2 = cv2.imread('../../../images/m-004-14.png')
    img_beauty = cv2.imread('../../../images/m-004-a.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_beauty = cv2.cvtColor(img_beauty, cv2.COLOR_BGR2RGB)
    # Extract landmark indexes
    extractor = LandmarkExtractor("../../../models/shape_predictor_68_face_landmarks.dat")
    points1 = extractor.get_2d_landmarks(img1)
    points2 = extractor.get_2d_landmarks(img2)
    points3 = extractor.get_2d_landmarks(img_beauty)
    # Align faces
    aligner = aligner.FaceAligner(desired_face_width=img1.shape[0])
    img1 = aligner.align(img1, points1)
    img2 = aligner.align(img2, points2)
    img_beauty = aligner.align(img_beauty, points3)
    # New landmarks
    points1 = extractor.get_2d_landmarks(img1)
    points2 = extractor.get_2d_landmarks(img2)
    points3 = extractor.get_2d_landmarks(img_beauty)
    # Extract indexes from one of the two
    triangles_indexes = get_triangulations_indexes(img1, points1)
    triangles_points1 = utils.triangulation_indexes_to_points(points1, triangles_indexes)
    triangles_points2 = utils.triangulation_indexes_to_points(points2, triangles_indexes)
    triangles_points3 = utils.triangulation_indexes_to_points(points3, triangles_indexes)
    # Draw Delaunay
    image_delaunay1 = utils.draw_delaunay_from_triangles(img1, triangles_points1, (150, 0, 0))
    image_delaunay2 = utils.draw_delaunay_from_triangles(img2, triangles_points2, (150, 0, 0))
    image_delaunay3 = utils.draw_delaunay_from_triangles(img_beauty, triangles_points3, (150, 0, 0))
    images = [(image_delaunay1, 'Delaunay controlled'),
              (image_delaunay2, 'Delaunay genuine'),
              (image_delaunay3, 'Delaunay beautified')]
    mosaic = get_images_mosaic_with_label('Delaunay', images, 1, 3)
    mosaic.show()

    # Compute area
    mean_area_difference12 = measures.compute_mean_triangles_area(triangles_points1, triangles_points2)
    mean_area_difference13 = measures.compute_mean_triangles_area(triangles_points1, triangles_points3)
    print('Mean areas difference ctrl-genuine:', mean_area_difference12)
    print('Mean areas difference ctrl-beauty:', mean_area_difference13)

    # Compute centroid
    centroid_distances12 = measures.compute_mean_centroids_distances(triangles_points1, triangles_points2)
    centroid_distances13 = measures.compute_mean_centroids_distances(triangles_points1, triangles_points3)
    print('Centroid distances ctrl-genuine:', centroid_distances12)
    print('Centroid distances ctrl-beauty:', centroid_distances13)

    # Compute cosine similarity
    angles_distances12 = measures.compute_mean_angles_distances(triangles_points1, triangles_points2)
    angles_distances13 = measures.compute_mean_angles_distances(triangles_points1, triangles_points3)
    print('Angles cosine distances ctrl-genuine:', angles_distances12)
    print('Angles cosine distances ctrl-beauty:', angles_distances13)

    # Compute matrix
    affine_matrices_distances = measures.compute_mean_affine_matrices_distances(triangles_points1,
                                                                                triangles_points2,
                                                                                triangles_points3)
    print('Affine matrix distances:', affine_matrices_distances)
