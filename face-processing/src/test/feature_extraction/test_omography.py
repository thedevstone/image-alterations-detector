import cv2
import matplotlib.pyplot as plt

from feature_extraction.faces.face_extractor import FaceExtractor
from feature_extraction.faces.utils import draw_faces_bounding_boxes
from feature_extraction.landmarks.landmark_extractor import LandmarkExtractor
from feature_extraction.landmarks.utils import visualize_facial_landmarks_areas, visualize_facial_landmarks_points
from feature_extraction.omography.omography import apply_homography_from_landmarks
from feature_extraction.utils.plotting import get_images_mosaic

if __name__ == '__main__':
    landmark_extractor = LandmarkExtractor("../../shape_predictor_68_face_landmarks.dat")
    face_extractor = FaceExtractor()
    # Images
    img1 = cv2.imread('../../../images/img1.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('../../../images/img2.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # Bbox and landmarks
    bbox_1 = face_extractor.get_faces_bbox(img1)
    bbox_2 = face_extractor.get_faces_bbox(img2)
    landmarks_1 = landmark_extractor.get_2d_landmarks(img1, bbox_1[0])
    landmarks_2 = landmark_extractor.get_2d_landmarks(img2, bbox_2[0])
    # landmarks_dnn = landmark_extractor.get_2d_landmarks_dnn(img)
    # Homography
    img1_aligned = apply_homography_from_landmarks(img1, img2, landmarks_1, landmarks_2)
    # Drawing
    img1_with_bbox = draw_faces_bounding_boxes(img1, bbox_1)
    img1_with_landmarks_area = visualize_facial_landmarks_areas(img1, landmarks_1)
    img1_with_landmarks_points = visualize_facial_landmarks_points(img1, landmarks_1)
    overlay = cv2.addWeighted(img1_aligned, 0.7, img2, 0.7, 0)
    images_to_show = [(img1, "Img 1"),
                      (img2, "Img 2"),
                      (img1_with_bbox, "Bbox 1"),
                      (img1_with_landmarks_points, "Landmarks points 1"),
                      (img1_with_landmarks_area, "Landmarks area 1"),
                      (img1_aligned, "Aligned 1"),
                      (overlay, "Overlay")]
    mosaic = get_images_mosaic("Homography", images_to_show, rows=2, cols=4)
    plt.show()
