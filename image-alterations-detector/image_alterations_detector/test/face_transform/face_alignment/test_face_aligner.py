import cv2

from face_transform.face_alignment.face_aligner import FaceAligner
from file_system.path_utilities import get_image_path
from plotting.plotting import get_images_mosaic_with_label


def main():
    face_aligner = FaceAligner()

    # Images
    img1 = cv2.imread(get_image_path("img1.jpg"), cv2.IMREAD_COLOR)
    img1_rotated = cv2.imread(get_image_path('img1_rotated.png'), cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1_rotated = cv2.cvtColor(img1_rotated, cv2.COLOR_BGR2RGB)

    aligned_image1 = face_aligner.align(img1)
    aligned_image1_rotated = face_aligner.align(img1_rotated)
    images_to_show = [(img1, "Image"), (aligned_image1, "Aligned"),
                      (img1_rotated, "Rotated"), (aligned_image1_rotated, "Aligned rotated")]
    mosaic = get_images_mosaic_with_label("Alignment", images_to_show, rows=2, cols=2)
    mosaic.show()


if __name__ == '__main__':
    main()
