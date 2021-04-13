import cv2
import numpy as np


def apply_homography_from_landmarks(img_source: np.ndarray, img_dest: np.ndarray, points_source: np.ndarray,
                                    points_dest: np.ndarray) -> np.ndarray:
    """ Compute homography between the two list of points and apply the homography to first image

    :param img_source: source image
    :param img_dest: destination image
    :param points_source: source points
    :param points_dest: destination points
    :return: the transformed input image
    """
    homography, mask = cv2.findHomography(points_source, points_dest, cv2.RANSAC, 5.0)
    img_source_aligned = cv2.warpPerspective(img_source, homography, dsize=(img_dest.shape[1], img_dest.shape[0]))
    return img_source_aligned
