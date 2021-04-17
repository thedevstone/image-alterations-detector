import cv2

from image_alterations_detector.face_morphology.landmarks_prediction.visualization import \
    visualize_facial_landmarks_points
from image_alterations_detector.face_transform.face_alignment.face_aligner import FaceAligner
from image_alterations_detector.file_system.path_utilities import get_image_path
from image_alterations_detector.plotting.plotting import get_images_mosaic_with_label


def main():
    face_aligner = FaceAligner()

    # Images
    img1 = cv2.imread(get_image_path("img1.jpg"), cv2.IMREAD_COLOR)
    img1_rotated = cv2.imread(get_image_path('img1_rotated.jpeg'), cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_rotated = cv2.cvtColor(img1_rotated, cv2.COLOR_BGR2RGB)

    aligned_image1, landmarks1 = face_aligner.align(img1)
    aligned_image1_rotated, landmarks1_rotated = face_aligner.align(img1_rotated)
    aligned_image1_landmarks = visualize_facial_landmarks_points(aligned_image1, landmarks1)
    aligned_image1_rotated_landmarks = visualize_facial_landmarks_points(aligned_image1_rotated, landmarks1_rotated)
    images_to_show = [(img1, "Image"), (aligned_image1, "Aligned"), (aligned_image1_landmarks, "Landmarks"),
                      (img1_rotated, "Rotated"), (aligned_image1_rotated, "Aligned rotated"),
                      (aligned_image1_rotated_landmarks, "Landmarks rotated")]
    mosaic = get_images_mosaic_with_label("Alignment", images_to_show, rows=2, cols=3)
    mosaic.show()


if __name__ == '__main__':
    main()
