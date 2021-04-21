from typing import Callable

import cv2
import numpy as np
from sklearn.preprocessing import minmax_scale

from image_alterations_detector.descriptors.triangle_descriptors.triangle_descriptors import \
    compute_triangle_affine_matrix
from image_alterations_detector.face_morphology.landmarks_prediction.visualization import \
    visualize_facial_landmarks_points
from image_alterations_detector.face_morphology.landmarks_triangulation.conversions import \
    triangulation_indexes_to_points
from image_alterations_detector.face_morphology.landmarks_triangulation.manage_triangulation import load_triangulation
from image_alterations_detector.face_transform.face_alignment.face_aligner import FaceAligner


def draw_delaunay_alterations(source_image, dest_image, animate=True,
                              show_function: Callable[[np.ndarray, np.ndarray], None] = None) -> np.ndarray:
    aligner = FaceAligner(desired_face_width=512)
    # Align
    source_image, source_image_landmarks = aligner.align(source_image)
    dest_image, dest_image_landmarks = aligner.align(dest_image)
    # Extract indexes from one image
    triangles_indexes = load_triangulation('triangulation.txt')
    source_image_landmarks = triangulation_indexes_to_points(source_image_landmarks, triangles_indexes)
    dest_image_landmarks = triangulation_indexes_to_points(dest_image_landmarks, triangles_indexes)
    # Sort triangles
    t1s, t2s = sort_altered_triangles(source_image_landmarks, dest_image_landmarks)
    # Output image
    source_out = source_image.copy()
    source_out = visualize_facial_landmarks_points(source_out, source_image_landmarks)
    dest_out = dest_image.copy()
    dest_out = visualize_facial_landmarks_points(dest_out, dest_image_landmarks)
    import colorsys

    for idx, t in enumerate(zip(t1s, t2s)):
        color_index = len(t1s) - idx
        rgb = colorsys.hsv_to_rgb(color_index / 300., 1.0, 1.0)
        delaunay_color = tuple([round(255 * x) for x in rgb])

        t1, t2 = t
        # Source
        tri_point_source1 = (t1[0], t1[1])
        tri_point_source2 = (t1[2], t1[3])
        tri_point_source3 = (t1[4], t1[5])

        cv2.line(source_out, tri_point_source1, tri_point_source2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(source_out, tri_point_source2, tri_point_source3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(source_out, tri_point_source3, tri_point_source1, delaunay_color, 1, cv2.LINE_AA, 0)

        # Dest
        tri_point_dest1 = (t2[0], t2[1])
        tri_point_dest2 = (t2[2], t2[3])
        tri_point_dest3 = (t2[4], t2[5])

        cv2.line(dest_out, tri_point_dest1, tri_point_dest2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(dest_out, tri_point_dest2, tri_point_dest3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(dest_out, tri_point_dest3, tri_point_dest1, delaunay_color, 1, cv2.LINE_AA, 0)

        if animate:
            show_function(source_out, dest_out)
            cv2.waitKey(200)

    if show_function:
        show_function(source_out, dest_out)
    return source_out


def sort_altered_triangles(source_image_landmarks, dest_image_landmarks):
    # Affine matrices
    matrices_distances = []
    source1_triangle_number = len(source_image_landmarks)
    source2_triangle_number = len(dest_image_landmarks)
    if source1_triangle_number != source2_triangle_number:
        raise AssertionError('Images have different number of triangles')
    for t1, t2 in zip(source_image_landmarks, dest_image_landmarks):
        affine_matrix = compute_triangle_affine_matrix(t1, t2)
        matrices_distances.append(affine_matrix.flatten())
    # Normalize affine matrices
    matrices_distances = np.array(matrices_distances)
    matrices_distances = minmax_scale(matrices_distances)
    # Computing mean matrix
    mean_matrix = matrices_distances.mean(axis=0)
    # Preparing for sort according to distance
    distances = []
    t1s = []
    t2s = []
    # Computing distance between matrices and mean matrix
    for matrix, t1, t2 in zip(matrices_distances, source_image_landmarks, dest_image_landmarks):
        dist = np.linalg.norm(matrix - mean_matrix)
        distances.append(dist)
        t1s.append(tuple(t1))
        t2s.append(tuple(t2))
    # Sorting lists according to distances
    t1s = [x for y, x in sorted(zip(distances, t1s))]
    t2s = [x for y, x in sorted(zip(distances, t2s))]
    distances.sort()
    t1s = np.array(t1s)
    t2s = np.array(t2s)
    return t1s, t2s
