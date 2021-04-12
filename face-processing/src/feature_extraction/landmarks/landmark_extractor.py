import dlib
import face_alignment
import numpy as np

from feature_extraction.utils.conversions import shape_to_np


# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
class LandmarkExtractor:
    def __init__(self, dat_file):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dat_file)
        self.deep_predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                           flip_input=False,
                                                           device='cpu')

    def get_2d_landmarks(self, img, rect):
        """
        Get 2d dlib's landmarks

        :param img: the image
        :param rect: the region of extraction
        :return: a list of landmark parts
        """
        landmarks = self.predictor(img, rect)
        landmarks_2d = shape_to_np(landmarks)
        return landmarks_2d

    def get_2d_landmarks_dnn(self, img):
        """
        Get 2d neural net landmarks

        :param img: the image
        :return: a numpy landmark array
        """
        landmarks = self.deep_predictor.get_landmarks(img)
        return np.array(landmarks[0], dtype='int')
