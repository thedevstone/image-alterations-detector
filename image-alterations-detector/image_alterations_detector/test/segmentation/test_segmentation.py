import os

import cv2

from image_alterations_detector.file_system.path_utilities import ROOT_DIR
from image_alterations_detector.plotting.plotting import get_images_mosaic_with_label
from image_alterations_detector.segmentation.face_segmenter import FaceSegmenter, denormalize_and_convert_rgb
from image_alterations_detector.utils.image_utils import load_img

if __name__ == '__main__':
    face_segmenter = FaceSegmenter()
    segmented = face_segmenter.segment_images_keep_aspect_ratio(
        [load_img('franco1.png'),
         load_img('franco2.png'),
         load_img('franco3.png'),
         load_img('franco4.bmp')
         ])
    rgb_images = denormalize_and_convert_rgb(segmented)
    cv2.imwrite(os.path.join(ROOT_DIR, 'images', 'franco-segmented1.png'),
                cv2.cvtColor(rgb_images[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(ROOT_DIR, 'images', 'franco-segmented2.png'),
                cv2.cvtColor(rgb_images[1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(ROOT_DIR, 'images', 'franco-segmented3.png'),
                cv2.cvtColor(rgb_images[2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(ROOT_DIR, 'images', 'franco-segmented4.png'),
                cv2.cvtColor(rgb_images[3], cv2.COLOR_RGB2BGR))
    get_images_mosaic_with_label('Segmented',
                                 [(rgb_images[0], 'img1'),
                                  (rgb_images[1], 'img2'),
                                  (rgb_images[2], 'img3'),
                                  (rgb_images[3], 'img4')
                                  ], 2, 2).show()
    # print('IOU:', compute_general_iou(segmented[0], segmented[1]))
    #
    # print('Single IOU:', compute_iou_per_mask(segmented[0], segmented[1]))
