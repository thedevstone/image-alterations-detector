import os
from typing import Tuple, List

import albumentations as a
import cv2
import numpy as np

transform_distort = a.Compose([
    a.ImageCompression(quality_lower=39, quality_upper=40, p=0.4),
    a.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True, p=0.2),
    a.RandomBrightnessContrast(p=0.3),
])

transform_beauty = a.Compose([
    a.RandomBrightnessContrast(p=0.3),
])


def augment_distort(img: np.ndarray):
    transformed = transform_distort(image=img)
    transformed_image = transformed["image"]
    return transformed_image


def augment_beauty(img: np.ndarray):
    transformed = transform_beauty(image=img)
    transformed_image = transformed["image"]
    return transformed_image


def load_altered_dataset_beauty(path_to_dataset, images_to_load=60) -> Tuple[
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
    # Face operations
    def exists(path):
        return os.path.exists(path)

    def load_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    genuine_dir = os.path.join(path_to_dataset, 'Genuine')
    beautified_dir = os.path.join(path_to_dataset, 'Beautified')

    genuine_1 = []
    genuine_14 = []
    genuine_up_sample1 = []
    genuine_up_sample2 = []
    beauty_a = []
    beauty_b = []
    beauty_c = []
    for gender in ['m', 'w']:
        for i in range(1, images_to_load):
            number = str(i).zfill(3)
            image_1_path = os.path.join(genuine_dir, '{}-{}-1.png'.format(gender, number))
            image_14_path = os.path.join(genuine_dir, '{}-{}-14.png'.format(gender, number))
            image_beauty_a_path = os.path.join(beautified_dir, 'a', '{}-{}-a.jpg'.format(gender, number))
            image_beauty_b_path = os.path.join(beautified_dir, 'b', '{}-{}-b.jpg'.format(gender, number))
            image_beauty_c_path = os.path.join(beautified_dir, 'c', '{}-{}-c.jpg'.format(gender, number))
            if exists(image_1_path) and exists(image_14_path):
                image_14 = load_image(image_14_path)
                genuine_1.append(load_image(image_1_path))
                genuine_14.append(image_14)
                genuine_up_sample1.append(augment_beauty(image_14))
                genuine_up_sample2.append(augment_beauty(image_14))
                beauty_a.append(load_image(image_beauty_a_path))
                beauty_b.append(load_image(image_beauty_b_path))
                beauty_c.append(load_image(image_beauty_c_path))

    # images_to_show = [genuine_1[0], genuine_5[0], genuine_14[0], beauty_a[0], beauty_b[0], beauty_c[0]]
    # mosaic = get_images_mosaic_no_labels('Dataset', images_to_show, 2, 3)
    # mosaic.show()
    return (genuine_1, genuine_14, genuine_up_sample1, genuine_up_sample2), (beauty_a, beauty_b, beauty_c)


def load_altered_dataset_distortion(path_to_dataset, images_to_load=60) -> Tuple[
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
    # Face operations
    def exists(path):
        return os.path.exists(path)

    def load_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    genuine_dir = os.path.join(path_to_dataset, 'Genuine')
    altered_barrel_dir = os.path.join(path_to_dataset, '0.14', 'Barrel')
    altered_pincushion_dir = os.path.join(path_to_dataset, '0.14', 'Pincushion')
    altered_decr_dir = os.path.join(path_to_dataset, '0.14', 'VDecr')
    altered_incr_dir = os.path.join(path_to_dataset, '0.14', 'VIncr')

    genuine_1 = []
    genuine_14 = []
    genuine_up_sample1 = []
    genuine_up_sample2 = []
    genuine_up_sample3 = []
    barrel = []
    pincushion = []
    decr = []
    incr = []
    for gender in ['m', 'w']:
        for i in range(1, images_to_load):
            number = str(i).zfill(3)
            image_1_path = os.path.join(genuine_dir, '{}-{}-1.png'.format(gender, number))
            image_14_path = os.path.join(genuine_dir, '{}-{}-14.png'.format(gender, number))
            image_barrel_path = os.path.join(altered_barrel_dir, '{}-{}-14.png'.format(gender, number))
            image_pincushion_path = os.path.join(altered_pincushion_dir, '{}-{}-14.png'.format(gender, number))
            image_decr_path = os.path.join(altered_decr_dir, '{}-{}-14.png'.format(gender, number))
            image_incr_path = os.path.join(altered_incr_dir, '{}-{}-14.png'.format(gender, number))
            if exists(image_1_path) and exists(image_14_path):
                image_14 = load_image(image_14_path)
                genuine_1.append(load_image(image_1_path))
                genuine_14.append(image_14)
                genuine_up_sample1.append(augment_distort(image_14))
                genuine_up_sample2.append(augment_distort(image_14))
                genuine_up_sample3.append(augment_distort(image_14))
                barrel.append(load_image(image_barrel_path))
                pincushion.append(load_image(image_pincushion_path))
                decr.append(load_image(image_decr_path))
                incr.append(load_image(image_incr_path))

    # images_to_show = [genuine_1[0], genuine_14[0], genuine_up_sample1[0], genuine_up_sample2[0], genuine_up_sample3[0],
    #                   barrel[0], pincushion[0], decr[0], incr[0]]
    # mosaic = get_images_mosaic_no_labels('Dataset', images_to_show, 3, 3)
    # mosaic.show()
    return (genuine_1, genuine_14, genuine_up_sample1, genuine_up_sample2, genuine_up_sample3), (
        barrel, pincushion, decr, incr)


if __name__ == '__main__':
    dataset_path = '/Users/luca/Desktop/altered'
    load_altered_dataset_distortion(dataset_path, 2)
