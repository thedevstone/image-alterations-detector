from typing import List

import numpy as np

from image_alterations_detector.face_transform.face_alignment.face_aligner import FaceAligner
from image_alterations_detector.file_system.path_utilities import get_model_path
from image_alterations_detector.segmentation.configuration.keras_backend import set_keras_backend


def segment_images(images: List[np.ndarray], convert_to_rgb=False):
    set_keras_backend()
    import image_alterations_detector.segmentation.configuration.color_configuration as color_conf
    import image_alterations_detector.segmentation.conversions as conversions
    import image_alterations_detector.segmentation.model as model

    # Configuration
    image_size = 256
    aligner = FaceAligner(desired_face_width=image_size)
    classes_to_segment = {'skin': True, 'nose': True, 'eye': True, 'brow': True, 'ear': True, 'mouth': True,
                          'hair': True, 'neck': True, 'cloth': False}
    classes_list, colors_values_list = color_conf.get_classes_colors(classes_to_segment)

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


def compute_iou(img1, img2):
    set_keras_backend()
    from segmentation_models.metrics import IOUScore
    import segmentation_models as sm
    # class_weight = np.array([0.29, 0.02, 0.00, 0.01, 0.01, 0.01, 0.33, 0.04, 0.28])
    segmented = segment_images([img1, img2])
    iou: IOUScore = sm.metrics.IOUScore(threshold=0.7)
    return iou(segmented[0], segmented[1])
