import os

import cv2

from feature_extraction.faces.face_extractor import FaceExtractor
from feature_extraction.landmarks.landmark_extractor import LandmarkExtractor
from feature_extraction.landmarks.utils import visualize_facial_landmarks_areas, visualize_facial_landmarks_points


def main():
    # WORKING DIRECTORY
    abspath = os.path.abspath(__file__)
    d_name = os.path.dirname(abspath)
    os.chdir(d_name)

    landmark_extractor = LandmarkExtractor()
    face_extractor = FaceExtractor()

    img = cv2.imread('img1.png', cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bbox = face_extractor.get_faces_bbox(img)
    landmarks = landmark_extractor.get_2d_landmarks(img, bbox[0])
    # landmarks_dnn = landmark_extractor.get_2d_landmarks_dnn(img)
    img_with_landmarks_area = visualize_facial_landmarks_areas(img, landmarks)
    img_with_landmarks_points = visualize_facial_landmarks_points(img, landmarks)
    cv2.imshow("Landmark", img_with_landmarks_area)
    cv2.imshow("Landmark points", img_with_landmarks_points)
    cv2.waitKey()


if __name__ == '__main__':
    main()
