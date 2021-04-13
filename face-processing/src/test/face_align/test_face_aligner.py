import cv2
import matplotlib.pyplot as plt

from face_align.face_aligner import FaceAligner
from feature_extraction.landmarks.landmark_extractor import LandmarkExtractor
from feature_extraction.utils.plotting import get_images_mosaic

if __name__ == '__main__':
    landmark_extractor = LandmarkExtractor("../../../models/shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner()

    # Images
    img1 = cv2.imread('../../../images/img1.jpg', cv2.IMREAD_COLOR)
    img1_rotated = cv2.imread('../../../images/img1_rotated.png', cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_rotated = cv2.cvtColor(img1_rotated, cv2.COLOR_BGR2RGB)

    shape1 = landmark_extractor.get_2d_landmarks(img1)
    shape1_rotated = landmark_extractor.get_2d_landmarks(img1_rotated)
    aligned_image1 = face_aligner.align(img1, shape1)
    aligned_image1_rotated = face_aligner.align(img1_rotated, shape1_rotated)
    images_to_show = [(img1, "Image"), (aligned_image1, "Aligned"),
                      (img1_rotated, "Rotated"), (aligned_image1_rotated, "Aligned rotated")]
    mosaic = get_images_mosaic("Alignment", images_to_show, rows=2, cols=2)
    plt.show()
