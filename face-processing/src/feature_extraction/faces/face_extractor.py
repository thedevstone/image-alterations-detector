import dlib


class FaceExtractor():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def get_faces_bbox(self, img):
        rects = self.detector(img)
        return rects
