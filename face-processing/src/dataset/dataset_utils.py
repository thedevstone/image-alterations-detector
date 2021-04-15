import os
from typing import Tuple, List

import cv2
import numpy as np


def load_altered_dataset(path_to_dataset) -> Tuple[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
                                                   Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
    def exists(path):
        return os.path.exists(path)

    def load_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # # equalize the histogram of the Y channel
        # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # # convert the YUV image back to RGB format
        # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    genuine_dir = os.path.join(path_to_dataset, 'Genuine')
    beautified_dir = os.path.join(path_to_dataset, 'Beautified')

    genuine_1 = []
    genuine_5 = []
    genuine_14 = []
    beauty_a = []
    beauty_b = []
    beauty_c = []
    for gender in ['m', 'w']:
        for i in range(1, 60):
            number = str(i).zfill(3)
            image_1_path = os.path.join(genuine_dir, '{}-{}-1.png'.format(gender, number))
            image_5_path = os.path.join(genuine_dir, '{}-{}-5.png'.format(gender, number))
            image_14_path = os.path.join(genuine_dir, '{}-{}-14.png'.format(gender, number))
            image_beauty_a_path = os.path.join(beautified_dir, 'a', '{}-{}-a.jpg'.format(gender, number))
            image_beauty_b_path = os.path.join(beautified_dir, 'b', '{}-{}-b.jpg'.format(gender, number))
            image_beauty_c_path = os.path.join(beautified_dir, 'c', '{}-{}-c.jpg'.format(gender, number))
            if exists(image_1_path) and exists(image_14_path) and exists(image_5_path):
                genuine_1.append(load_image(image_1_path))
                genuine_5.append(load_image(image_5_path))
                genuine_14.append(load_image(image_14_path))
                beauty_a.append(load_image(image_beauty_a_path))
                beauty_b.append(load_image(image_beauty_b_path))
                beauty_c.append(load_image(image_beauty_c_path))

    # images_to_show = [genuine_1[0], genuine_5[0], genuine_14[0], beauty_a[0], beauty_b[0], beauty_c[0]]
    # mosaic = get_images_mosaic_no_labels('Dataset', images_to_show, 2, 3)
    # mosaic.show()
    return (genuine_1, genuine_5, genuine_14), (beauty_a, beauty_b, beauty_c)
