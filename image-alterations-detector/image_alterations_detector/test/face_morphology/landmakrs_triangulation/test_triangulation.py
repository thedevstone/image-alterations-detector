import cv2

from face_morphology.landmarks_prediction.landmark_predictor import LandmarkPredictor
from face_morphology.landmarks_triangulation.conversions import triangulation_indexes_to_points
from face_morphology.landmarks_triangulation.delaunay import compute_triangulation_indexes
from face_morphology.landmarks_triangulation.visualization import draw_delaunay_from_triangles_points
from face_transform.face_alignment.face_aligner import FaceAligner
from file_system.path_utilities import get_image_path
from plotting.plotting import get_images_mosaic_with_label


def main():
    # Load images
    img1 = cv2.imread(get_image_path('m-002-1.png'))
    img2 = cv2.imread(get_image_path('m-002-14.png'))
    img_beauty = cv2.imread(get_image_path('m-002-a.jpg'))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_beauty = cv2.cvtColor(img_beauty, cv2.COLOR_BGR2RGB)
    # Extract landmark indexes
    extractor = LandmarkPredictor()
    # Align face_detection
    aligner = FaceAligner(desired_face_width=img1.shape[0])
    img1 = aligner.align(img1)
    img2 = aligner.align(img2)
    img_beauty = aligner.align(img_beauty)
    # New landmarks_prediction
    points1 = extractor.get_2d_landmarks(img1)
    points2 = extractor.get_2d_landmarks(img2)
    points3 = extractor.get_2d_landmarks(img_beauty)
    # Extract indexes from one of the two
    triangles_indexes = compute_triangulation_indexes(img1, points1)
    # Triangulation on all images
    triangles_points1 = triangulation_indexes_to_points(points1, triangles_indexes)
    triangles_points2 = triangulation_indexes_to_points(points2, triangles_indexes)
    triangles_points3 = triangulation_indexes_to_points(points3, triangles_indexes)
    # Draw Delaunay
    image_delaunay1 = draw_delaunay_from_triangles_points(img1, triangles_points1, (150, 0, 0))
    image_delaunay2 = draw_delaunay_from_triangles_points(img2, triangles_points2, (150, 0, 0))
    image_delaunay3 = draw_delaunay_from_triangles_points(img_beauty, triangles_points3, (150, 0, 0))
    # Prepare mosaic
    images = [(image_delaunay1, 'Delaunay genuine'),
              (image_delaunay2, 'Delaunay controlled'),
              (image_delaunay3, 'Delaunay beautified')]
    mosaic = get_images_mosaic_with_label('Delaunay', images, 1, 3)
    mosaic.show()


if __name__ == '__main__':
    main()
