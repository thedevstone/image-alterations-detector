from typing import List, Tuple

import numpy as np
import tensorflow as tf
from

from image_alterations_detector.face_transform.face_alignment.face_aligner import FaceAligner
from image_alterations_detector.file_system.path_utilities import get_model_path
from image_alterations_detector.segmentation.configuration.color_configuration import get_classes_list
from image_alterations_detector.segmentation.configuration.keras_backend import set_keras_backend

CLASSES_TO_SEGMENT = {'skin': True, 'nose': True, 'eye': True, 'brow': True, 'ear': True, 'mouth': True,
                      'hair': True, 'neck': True, 'cloth': False}


def segment_images(images: List[np.ndarray], convert_to_rgb=False):
    """ Perform segmentation on a list of images

    :param images: the list of images
    :param convert_to_rgb: if convert to rgb
    :return: the list of segmented masks (converted in rgb if selected)
    """
    set_keras_backend()
    import image_alterations_detector.segmentation.configuration.color_configuration as color_conf
    import image_alterations_detector.segmentation.conversions as conversions
    import image_alterations_detector.segmentation.model as model

    # Configuration
    image_size = 256
    aligner = FaceAligner(desired_face_width=image_size)
    colors_values_list = color_conf.get_classes_colors(CLASSES_TO_SEGMENT)
    # Load the model
    inference_model = model.load_model(get_model_path('unet.h5'))
    # Output images
    predicted_images = []
    # Images
    for img in images:
        img, landmarks = aligner.align(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2])).astype('float')
        img1_normalized = img / 255.0
        images_predicted = inference_model.predict(img1_normalized)
        image_predicted = images_predicted[0]
        if convert_to_rgb:
            image_predicted = conversions.denormalize(image_predicted)
            image_predicted = conversions.mask_channels_to_rgb(image_predicted, 8, image_size, colors_values_list)
        predicted_images.append(image_predicted)
    return predicted_images


def compute_general_iou(img1, img2) -> float:
    """ Compute the IOU value for the images

    :param img1: the source image
    :param img2: the destination image
    :return: the IOU value
    """
    set_keras_backend()
    from segmentation_models.metrics import IOUScore
    import segmentation_models as sm
    # class_weight = np.array([0.29, 0.02, 0.00, 0.01, 0.01, 0.01, 0.33, 0.04, 0.28])
    iou: IOUScore = sm.metrics.IOUScore(threshold=0.7)
    segmented = segment_images([img1, img2])
    return iou(segmented[0], segmented[1])


def compute_iou_per_mask(img1, img2) -> List[Tuple[str, float]]:
    """ Compute the IOU on all masks

    :param img1: the source image
    :param img2: the destination image
    :return: the list of all IOU for each mask
    """
    set_keras_backend()
    from segmentation_models.metrics import IOUScore
    import segmentation_models as sm
    # class_weight = np.array([0.29, 0.02, 0.00, 0.01, 0.01, 0.01, 0.33, 0.04, 0.28])
    iou: IOUScore = sm.metrics.IOUScore(threshold=0.7)
    segmented = segment_images([img1, img2])
    segmented1 = segmented[0]
    segmented2 = segmented[1]
    iou_values = []
    for idx, clazz in enumerate(get_classes_list(CLASSES_TO_SEGMENT)):
        mask1 = np.expand_dims(segmented1[:, :, idx], 2)
        mask2 = np.expand_dims(segmented2[:, :, idx], 2)
        iou_mask_tensor: tf.Tensor = iou(mask1, mask2)
        iou_mask_value = tf.keras.backend.get_value(iou_mask_tensor)
        iou_values.append((clazz, iou_mask_value))
    return iou_values
