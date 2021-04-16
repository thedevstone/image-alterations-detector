import cv2
import dlib
import numpy as np

from descriptors.texture_descriptors.local_binary_pattern import LocalBinaryPattern
from face_morphology.face_detection.conversions import rect_to_bounding_box
from face_morphology.face_detection.face_detector import FaceDetector


def compute_face_lbp_difference(source_img: np.ndarray, dest_img: np.ndarray, detector: FaceDetector,
                                lpb_descriptor: LocalBinaryPattern) -> np.ndarray:
    """ Compute local binary pattern descriptor for source and destination image

    **Descriptor**

    The descriptor is obtained concatenating the source and destination flattened lbp histograms. [lbp_hist1, lbp_hist2]

    **Descriptor Idea**

    The idea is that image alteration and beautification would affect image texture


    :param source_img: the source image
    :param dest_img: the destination image
    :param detector: the face detector
    :param lpb_descriptor: the local binary pattern extractor
    :return:
    """
    bbox_source: dlib.rectangle = detector.get_faces_bbox(source_img)[0]
    bbox_dest: dlib.rectangle = detector.get_faces_bbox(dest_img)[0]
    (x1, y1, w1, h1) = rect_to_bounding_box(bbox_source)
    (x2, y2, w2, h2) = rect_to_bounding_box(bbox_dest)
    source_img_crop = source_img.copy()[y1:y1 + h1, x1:x1 + w1]
    dest_img_crop = dest_img.copy()[y2:y2 + h2, x2:x2 + w2]
    source_img_crop = cv2.cvtColor(source_img_crop, cv2.COLOR_RGB2GRAY)
    dest_img_crop = cv2.cvtColor(dest_img_crop, cv2.COLOR_RGB2GRAY)
    lbp_source = lpb_descriptor.describe(source_img_crop)
    lbp_dest = lpb_descriptor.describe(dest_img_crop)
    lbp_complete = np.concatenate([lbp_source.flatten(), lbp_dest.flatten()])
    return lbp_complete
