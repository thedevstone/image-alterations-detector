import dlib
import numpy as np


class FaceExtractor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def get_faces_bbox(self, img: np.ndarray) -> dlib.rectangles:
        """ Extract face bounding boxes

        :param img: the input image
        :return: the face bboxes
        """
        rects = self.detector(img)
        return rects
