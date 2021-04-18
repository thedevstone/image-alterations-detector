from typing import Tuple

import numpy as np
from sklearn.preprocessing import normalize

from image_alterations_detector.dataset.altered_dataset.altered_dataset_utils import load_altered_dataset
from image_alterations_detector.descriptors.double_image_alteration_descriptors.shape_transform_descriptor import \
    compute_affine_matrices_descriptor
from image_alterations_detector.descriptors.double_image_alteration_descriptors.texture_alteration_descriptor import \
    compute_face_lbp_difference
from image_alterations_detector.descriptors.double_image_alteration_descriptors.triangles_measures_alteration_descriptor import \
    compute_mean_triangles_area_differences_descriptor
from image_alterations_detector.descriptors.texture_descriptors.local_binary_pattern import LocalBinaryPattern
from image_alterations_detector.face_morphology.face_detection.face_detector import FaceDetector
from image_alterations_detector.face_morphology.landmarks_prediction.landmark_predictor import LandmarkPredictor
from image_alterations_detector.face_morphology.landmarks_triangulation.conversions import \
    triangulation_indexes_to_points
from image_alterations_detector.face_morphology.landmarks_triangulation.delaunay import compute_triangulation_indexes
from image_alterations_detector.face_transform.face_alignment.face_aligner import FaceAligner


def compute_two_image_descriptors(source_image, dest_image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    extractor = LandmarkPredictor()
    detector = FaceDetector()
    aligner = FaceAligner(desired_face_width=512)
    # Descriptor
    lbp = LocalBinaryPattern(24, 8)
    # Extract indexes from one image
    points = extractor.get_2d_landmarks(source_image)
    triangles_indexes = compute_triangulation_indexes(source_image, points)
    # Align
    source_image, source_image_landmarks = aligner.align(source_image)
    dest_image, dest_image_landmarks = aligner.align(dest_image)
    # Extract unique indexes
    source_image_landmarks = triangulation_indexes_to_points(source_image_landmarks, triangles_indexes)
    dest_image_landmarks = triangulation_indexes_to_points(dest_image_landmarks, triangles_indexes)  # repeated
    # Descriptors
    mean_area = compute_mean_triangles_area_differences_descriptor(source_image_landmarks, dest_image_landmarks)
    affine_matrices = compute_affine_matrices_descriptor(source_image_landmarks, dest_image_landmarks)
    lbp = compute_face_lbp_difference(source_image, dest_image, detector, lbp)
    mean_area = np.array(mean_area).astype('float32')
    affine_matrices = np.array(affine_matrices).astype('float32')
    lbp = np.array(lbp).astype('float32')
    # Normalize
    mean_area_descriptors = normalize(np.expand_dims(mean_area, 0), norm='max')
    matrices_descriptors = normalize(np.expand_dims(affine_matrices, 0), norm='max')
    lbp_descriptors = normalize(np.expand_dims(lbp, 0), norm='max')
    return mean_area_descriptors, matrices_descriptors, lbp_descriptors


def compute_altered_descriptors(dataset_path, images_to_load=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                            np.ndarray]:
    # Descriptors
    mean_area_descriptors = []
    matrices_descriptors = []
    lbp_descriptors = []
    labels = []
    # Load the dataset
    genuine, altered = load_altered_dataset(dataset_path)
    genuine_1, genuine_5, genuine_14 = genuine
    beauty_a, beauty_b, beauty_c = altered
    # Face operations
    extractor = LandmarkPredictor()
    detector = FaceDetector()
    aligner = FaceAligner(desired_face_width=512)
    # Descriptor
    lbp = LocalBinaryPattern(24, 8)
    # Extract indexes from one image
    points = extractor.get_2d_landmarks(genuine_1[0])
    triangles_indexes = compute_triangulation_indexes(genuine_1[0], points)
    for idx in range(0, len(genuine_1) if not images_to_load else images_to_load):
        img_genuine_1 = genuine_1[idx]
        img_genuine_5 = genuine_5[idx]
        img_genuine_14 = genuine_14[idx]
        img_beauty_a = beauty_a[idx]
        img_beauty_b = beauty_b[idx]
        img_beauty_c = beauty_c[idx]
        # Align face
        try:
            img_genuine_1, img_genuine_1_points = aligner.align(img_genuine_1)
            img_genuine_5, img_genuine_5_points = aligner.align(img_genuine_5)
            img_genuine_14, img_genuine_14_points = aligner.align(img_genuine_14)
            img_beauty_a, img_beauty_a_points = aligner.align(img_beauty_a)
            img_beauty_b, img_beauty_b_points = aligner.align(img_beauty_b)
            img_beauty_c, img_beauty_c_points = aligner.align(img_beauty_c)
        except IndexError:
            continue
        # Extract points from indexes
        img_genuine_1_tri_points = triangulation_indexes_to_points(img_genuine_1_points, triangles_indexes)
        img_genuine_5_tri_points = triangulation_indexes_to_points(img_genuine_5_points, triangles_indexes)
        img_genuine_14_tri_points = triangulation_indexes_to_points(img_genuine_14_points, triangles_indexes)
        img_beauty_a_tri_points = triangulation_indexes_to_points(img_beauty_a_points, triangles_indexes)
        img_beauty_b_tri_points = triangulation_indexes_to_points(img_beauty_b_points, triangles_indexes)
        img_beauty_c_tri_points = triangulation_indexes_to_points(img_beauty_c_points, triangles_indexes)
        # Compute mean area
        mean_area_difference1_14 = compute_mean_triangles_area_differences_descriptor(img_genuine_1_tri_points,
                                                                                      img_genuine_14_tri_points)
        mean_area_difference1_5 = compute_mean_triangles_area_differences_descriptor(img_genuine_1_tri_points,
                                                                                     img_genuine_5_tri_points)
        mean_area_difference1_a = compute_mean_triangles_area_differences_descriptor(img_genuine_1_tri_points,
                                                                                     img_beauty_a_tri_points)
        mean_area_difference1_b = compute_mean_triangles_area_differences_descriptor(img_genuine_1_tri_points,
                                                                                     img_beauty_b_tri_points)
        mean_area_difference1_c = compute_mean_triangles_area_differences_descriptor(img_genuine_1_tri_points,
                                                                                     img_beauty_c_tri_points)
        mean_area_descriptors.extend(
            [mean_area_difference1_14, mean_area_difference1_5, mean_area_difference1_a, mean_area_difference1_b,
             mean_area_difference1_c])
        # Matrix distances
        affine_matrices_distances1_14 = compute_affine_matrices_descriptor(img_genuine_1_tri_points,
                                                                           img_genuine_14_tri_points)
        affine_matrices_distances1_5 = compute_affine_matrices_descriptor(img_genuine_1_tri_points,
                                                                          img_genuine_5_tri_points)
        affine_matrices_distances1_a = compute_affine_matrices_descriptor(img_genuine_1_tri_points,
                                                                          img_beauty_a_tri_points)
        affine_matrices_distances1_b = compute_affine_matrices_descriptor(img_genuine_1_tri_points,
                                                                          img_beauty_b_tri_points)
        affine_matrices_distances1_c = compute_affine_matrices_descriptor(img_genuine_1_tri_points,
                                                                          img_beauty_c_tri_points)
        matrices_descriptors.extend(
            [affine_matrices_distances1_14, affine_matrices_distances1_5, affine_matrices_distances1_a,
             affine_matrices_distances1_b, affine_matrices_distances1_c])
        # LBP
        lbp_features1_14 = compute_face_lbp_difference(img_genuine_1, img_genuine_14, detector, lbp)
        lbp_features1_5 = compute_face_lbp_difference(img_genuine_1, img_genuine_5, detector, lbp)
        lbp_features1_a = compute_face_lbp_difference(img_genuine_1, img_beauty_a, detector, lbp)
        lbp_features1_b = compute_face_lbp_difference(img_genuine_1, img_beauty_b, detector, lbp)
        lbp_features1_c = compute_face_lbp_difference(img_genuine_1, img_beauty_c, detector, lbp)

        lbp_descriptors.extend([lbp_features1_14, lbp_features1_5, lbp_features1_a, lbp_features1_b, lbp_features1_c])

        # Setting labels
        labels.extend([0, 0, 1, 1, 1])

    mean_area_descriptors = np.array(mean_area_descriptors).astype('float32')
    matrices_descriptors = np.array(matrices_descriptors).astype('float32')
    lbp_descriptors = np.array(lbp_descriptors).astype('float32')
    # Convert labels
    labels = np.array(labels)

    # Normalize
    mean_area_descriptors = normalize(mean_area_descriptors, norm='max')
    matrices_descriptors = normalize(matrices_descriptors, norm='max')
    lbp_descriptors = normalize(lbp_descriptors, norm='max')

    return mean_area_descriptors, matrices_descriptors, lbp_descriptors, labels
