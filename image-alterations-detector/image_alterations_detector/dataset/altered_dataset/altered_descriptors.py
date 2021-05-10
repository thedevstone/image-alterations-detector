from typing import Tuple

import numpy as np

from image_alterations_detector.dataset.altered_dataset.altered_dataset_utils import load_altered_dataset_beauty, \
    load_altered_dataset_distortion
from image_alterations_detector.descriptors.double_image_alteration_descriptors.shape_transform_descriptor import \
    compute_affine_matrices_descriptor
from image_alterations_detector.descriptors.double_image_alteration_descriptors.texture_alteration_descriptor import \
    compute_face_lbp_difference
from image_alterations_detector.descriptors.double_image_alteration_descriptors.triangles_measures_alteration_descriptor import \
    compute_mean_triangles_area_differences_descriptor, compute_mean_triangles_angles_distances_descriptor
from image_alterations_detector.descriptors.texture_descriptors.local_binary_pattern import LocalBinaryPattern
from image_alterations_detector.face_morphology.face_detection.face_detector import FaceDetector
from image_alterations_detector.face_morphology.landmarks_triangulation.conversions import \
    triangulation_indexes_to_points
from image_alterations_detector.face_morphology.landmarks_triangulation.manage_triangulation import load_triangulation
from image_alterations_detector.face_transform.face_alignment.face_aligner import FaceAligner

ANGLES_DIM = 339
AFFINE_MATRICES_DIM = 678
LBP_DIM = 52

# Face operations
detector = FaceDetector()
aligner = FaceAligner(desired_face_width=512)
lbp = LocalBinaryPattern(24, 8)


def compute_two_image_descriptors_beauty(source_image, dest_image) -> np.ndarray:
    # Align
    source_image, source_image_landmarks = aligner.align(source_image)
    dest_image, dest_image_landmarks = aligner.align(dest_image)
    # Descriptors
    lbp_descriptor = compute_face_lbp_difference(source_image, dest_image, detector, lbp)
    # Convert to float
    lbp_descriptor = np.array(lbp_descriptor).astype('float32')
    return lbp_descriptor


def compute_two_image_descriptors_distortion(source_image, dest_image, triangles_indexes) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Align
    source_image, source_image_landmarks = aligner.align(source_image)
    dest_image, dest_image_landmarks = aligner.align(dest_image)
    # Extract unique indexes
    source_image_landmarks = triangulation_indexes_to_points(source_image_landmarks, triangles_indexes)
    dest_image_landmarks = triangulation_indexes_to_points(dest_image_landmarks, triangles_indexes)  # repeated
    # Descriptors
    angles = compute_mean_triangles_angles_distances_descriptor(source_image_landmarks, dest_image_landmarks)
    mean_area = compute_mean_triangles_area_differences_descriptor(source_image_landmarks, dest_image_landmarks)
    affine_matrices = compute_affine_matrices_descriptor(source_image_landmarks, dest_image_landmarks)
    # Convert to float
    angles = np.array(angles).astype('float32')
    mean_area = np.array(mean_area).astype('float32')
    affine_matrices = np.array(affine_matrices).astype('float32')
    return angles, mean_area, affine_matrices


def compute_altered_descriptors_distortion(dataset_path, images_to_load=None) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Descriptors
    angles_descriptors = []
    mean_area_descriptors = []
    matrices_descriptors = []
    labels = []
    # Load the dataset
    genuine, altered = load_altered_dataset_distortion(dataset_path, images_to_load)
    genuine_1, genuine_14, genuine_up_sample1, genuine_up_sample2, genuine_up_sample3 = genuine
    barrel, pincushion, decr, incr = altered
    # Extract indexes from one image
    triangles_indexes = load_triangulation('triangulation.txt')
    for idx in range(0, images_to_load):
        img_genuine_1 = genuine_1[idx]
        img_genuine_14 = genuine_14[idx]
        img_up1 = genuine_up_sample1[idx]
        img_up2 = genuine_up_sample2[idx]
        img_up3 = genuine_up_sample3[idx]
        img_barrel = barrel[idx]
        img_pincushion = pincushion[idx]
        img_decr = decr[idx]
        img_incr = incr[idx]
        # Align face
        try:
            angles_1_14, mean_area_1_14, affine_matrices_1_14 = compute_two_image_descriptors_distortion(
                img_genuine_1, img_genuine_14, triangles_indexes)
            angles_1_up1, mean_area_1_up1, affine_matrices_1_up1 = compute_two_image_descriptors_distortion(
                img_genuine_1, img_up1, triangles_indexes)
            angles_1_up2, mean_area_1_up2, affine_matrices_1_up2 = compute_two_image_descriptors_distortion(
                img_genuine_1, img_up2, triangles_indexes)
            angles_1_up3, mean_area_1_up3, affine_matrices_1_up3 = compute_two_image_descriptors_distortion(
                img_genuine_1, img_up3, triangles_indexes)

            angles_barrel, mean_area_barrel, affine_matrices_barrel = compute_two_image_descriptors_distortion(
                img_genuine_1, img_barrel, triangles_indexes)
            angles_pincushion, mean_area_pincushion, affine_matrices_pincushion = compute_two_image_descriptors_distortion(
                img_genuine_1, img_pincushion, triangles_indexes)
            angles_decr, mean_area_decr, affine_matrices_decr = compute_two_image_descriptors_distortion(
                img_genuine_1, img_decr, triangles_indexes)
            angles_incr, mean_area_incr, affine_matrices_incr = compute_two_image_descriptors_distortion(
                img_genuine_1, img_incr, triangles_indexes)
        except IndexError:
            continue
        angles_descriptors.extend([angles_1_14, angles_1_up1, angles_1_up2, angles_1_up3,
                                   angles_barrel, angles_pincushion, angles_decr, angles_incr])
        mean_area_descriptors.extend([mean_area_1_14, mean_area_1_up1, mean_area_1_up2, mean_area_1_up3,
                                      mean_area_barrel, mean_area_pincushion, mean_area_decr, mean_area_incr])
        matrices_descriptors.extend(
            [affine_matrices_1_14, affine_matrices_1_up1, affine_matrices_1_up2, affine_matrices_1_up3,
             affine_matrices_barrel, affine_matrices_pincushion, affine_matrices_decr, affine_matrices_incr])
        labels.extend([1, 1, 1, 1, 0, 0, 0, 0])

    angles_descriptors = np.array(angles_descriptors).astype('float32')
    mean_area_descriptors = np.array(mean_area_descriptors).astype('float32')
    matrices_descriptors = np.array(matrices_descriptors).astype('float32')
    # Convert labels
    labels = np.array(labels)
    return angles_descriptors, mean_area_descriptors, matrices_descriptors, labels


def compute_altered_descriptors_beauty(dataset_path, images_to_load=None) -> Tuple[np.ndarray, np.ndarray]:
    # Descriptors
    lbp_descriptors = []
    labels = []
    # Load the dataset
    genuine, altered = load_altered_dataset_beauty(dataset_path, images_to_load)
    genuine_1, genuine_14 = genuine
    beauty_a, beauty_b, beauty_c = altered
    for idx in range(0, images_to_load):
        img_genuine_1, landmarks_1 = aligner.align(genuine_1[idx])
        img_genuine_14, landmarks_14 = aligner.align(genuine_14[idx])
        img_beauty_a, landmarks_a = aligner.align(beauty_a[idx])
        img_beauty_b, landmarks_b = aligner.align(beauty_b[idx])
        img_beauty_c, landmarks_c = aligner.align(beauty_c[idx])

        # LBP
        lbp_features1_14 = compute_face_lbp_difference(img_genuine_1, img_genuine_14, detector, lbp)
        lbp_features1_a = compute_face_lbp_difference(img_genuine_1, img_beauty_a, detector, lbp)
        lbp_features1_b = compute_face_lbp_difference(img_genuine_1, img_beauty_b, detector, lbp)
        lbp_features1_c = compute_face_lbp_difference(img_genuine_1, img_beauty_c, detector, lbp)

        lbp_descriptors.extend([lbp_features1_14, lbp_features1_a, lbp_features1_b, lbp_features1_c])

        # Setting labels
        labels.extend([1, 0, 0, 0])

    lbp_descriptors = np.array(lbp_descriptors).astype('float32')
    # Convert labels
    labels = np.array(labels)
    return lbp_descriptors, labels
