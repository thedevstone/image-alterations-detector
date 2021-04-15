import dlib
import face_alignment
import numpy as np

from feature_extraction.utils.conversions import landmarks_to_array


# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
class LandmarkPredictor:
    def __init__(self, dat_file, predictor_type='dlib'):
        """ Initialize the landmark predictor

        :param dat_file: path to the .dat model for dlib predictor
        :param predictor_type: the predictor type. 'dlib' for standard fast predictor or 'dnn' for slow deep predictor
        """
        self.predictor_type = predictor_type
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dat_file)
        self.deep_predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                                           device='cpu')

    def get_2d_landmarks(self, img: np.ndarray) -> np.ndarray:
        """
        Get 2d dlib's landmarks

        :param img: the image
        :return: a list of landmark parts
        """
        face_bbox = self.detector(img)
        if len(face_bbox) == 0:
            raise IndexError("No faces found has been found")
        if self.predictor_type == 'dlib':
            landmarks = self.predictor(img, face_bbox[0])
            landmarks_2d = landmarks_to_array(landmarks)
            return landmarks_2d
        else:
            landmarks = self.deep_predictor.get_landmarks(img)[0]
            return np.array(landmarks[0], dtype='int')

    def get_2d_landmarks_from_bbox(self, img: np.ndarray, rect: dlib.rectangle) -> np.ndarray:
        """
        Get 2d dlib's landmarks

        :param img: the image
        :param rect: the region of extraction
        :return: a list of landmark parts
        """
        landmarks = self.predictor(img, rect)
        landmarks_2d = landmarks_to_array(landmarks)
        return landmarks_2d
