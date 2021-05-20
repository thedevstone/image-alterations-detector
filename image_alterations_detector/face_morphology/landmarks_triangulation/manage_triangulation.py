import os.path
from os.path import exists

import cv2
import numpy as np

from image_alterations_detector.face_morphology.landmarks_triangulation.delaunay import compute_triangulation_indexes
from image_alterations_detector.face_transform.face_alignment.face_aligner import FaceAligner
from image_alterations_detector.file_system.path_utilities import get_image_path, get_folder_path_from_root, \
    get_model_path


def serialize_triangulation(image: np.ndarray, filename: str) -> None:
    """ Serialize the Delaunay triangulation on file

    :param image: the sample image
    :param filename: the filename
    """
    aligner = FaceAligner(desired_face_width=512)
    image, points = aligner.align(image)
    triangles_indexes = compute_triangulation_indexes(img, points)
    file_path = os.path.join(get_folder_path_from_root('models'), filename)
    write_mode = 'a' if not exists(file_path) else 'w'
    with open(file_path, write_mode) as file:
        for t in triangles_indexes:
            file.write('{} {} {}\n'.format(t[0], t[1], t[2]))


def load_triangulation(file_name) -> np.ndarray:
    """ Load the triangulation from file

    :param file_name: the triangulation file name
    :return: the numpy triangulation array
    """
    triangles_indexes = []
    # Read triangles from tri.txt
    with open(get_model_path(file_name)) as file:
        for line in file:
            x, y, z = line.split()
            x = int(x)
            y = int(y)
            z = int(z)
            triangles_indexes.append([x, y, z])
    triangles_indexes = np.array(triangles_indexes)
    return triangles_indexes


if __name__ == '__main__':
    img = cv2.imread(get_image_path('delaunay_sample.png'), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    serialize_triangulation(img, 'triangulation.txt')
    load_triangulation('triangulation.txt')
