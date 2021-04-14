import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from dataset.dataset_utils import load_altered_dataset
from face_align.face_aligner import FaceAligner
from feature_extraction.landmarks.landmark_extractor import LandmarkExtractor
from feature_extraction.triangulation import utils
from feature_extraction.triangulation.delaunay import get_triangulations_indexes
from measures.triangles_measures import compute_mean_triangles_area, compute_mean_centroids_distances, \
    compute_mean_angles_distances

if __name__ == '__main__':
    dataset_path = '/Users/luca/Desktop/altered'
    mean_area = []
    mean_area_labels = []
    centroids = []
    centroids_labels = []
    angles = []
    angles_labels = []

    dataset = []
    labels = []
    genuine, altered = load_altered_dataset(dataset_path)
    genuine_1, genuine_5, genuine_14 = genuine
    beauty_a, beauty_b, beauty_c = altered
    extractor = LandmarkExtractor("../../models/shape_predictor_68_face_landmarks.dat")
    aligner = FaceAligner(desired_face_width=genuine_1[0].shape[0])
    # Extract indexes from one of the two
    points = extractor.get_2d_landmarks(genuine_1[0])
    triangles_indexes = get_triangulations_indexes(genuine_1[0], points)
    for idx in range(0, len(genuine_1)):
        img_genuine_1 = genuine_1[idx]
        img_genuine_5 = genuine_5[idx]
        img_genuine_14 = genuine_14[idx]
        img_beauty_a = beauty_a[idx]
        img_beauty_b = beauty_b[idx]
        img_beauty_c = beauty_c[idx]
        # Extract landmark indexes
        img_genuine_1_points = extractor.get_2d_landmarks(img_genuine_1)
        img_genuine_5_points = extractor.get_2d_landmarks(img_genuine_5)
        img_genuine_14_points = extractor.get_2d_landmarks(img_genuine_14)
        img_beauty_a_points = extractor.get_2d_landmarks(img_beauty_a)
        img_beauty_b_points = extractor.get_2d_landmarks(img_beauty_b)
        img_beauty_c_points = extractor.get_2d_landmarks(img_beauty_c)
        # Align faces
        img_genuine_1 = aligner.align(img_genuine_1, img_genuine_1_points)
        img_genuine_5 = aligner.align(img_genuine_5, img_genuine_5_points)
        img_genuine_14 = aligner.align(img_genuine_14, img_genuine_14_points)
        img_beauty_a = aligner.align(img_beauty_a, img_beauty_a_points)
        img_beauty_b = aligner.align(img_beauty_b, img_beauty_b_points)
        img_beauty_c = aligner.align(img_beauty_c, img_beauty_c_points)
        # Extract landmark indexes
        img_genuine_1_points = extractor.get_2d_landmarks(img_genuine_1)
        img_genuine_5_points = extractor.get_2d_landmarks(img_genuine_5)
        img_genuine_14_points = extractor.get_2d_landmarks(img_genuine_14)
        img_beauty_a_points = extractor.get_2d_landmarks(img_beauty_a)
        img_beauty_b_points = extractor.get_2d_landmarks(img_beauty_b)
        img_beauty_c_points = extractor.get_2d_landmarks(img_beauty_c)
        # Extract indexes from one of the two
        img_genuine_1_tri_points = utils.triangulation_indexes_to_points(img_genuine_1_points, triangles_indexes)
        img_genuine_5_tri_points = utils.triangulation_indexes_to_points(img_genuine_5_points, triangles_indexes)
        img_genuine_14_tri_points = utils.triangulation_indexes_to_points(img_genuine_14_points, triangles_indexes)
        img_beauty_a_tri_points = utils.triangulation_indexes_to_points(img_beauty_a_points, triangles_indexes)
        img_beauty_b_tri_points = utils.triangulation_indexes_to_points(img_beauty_b_points, triangles_indexes)
        img_beauty_c_tri_points = utils.triangulation_indexes_to_points(img_beauty_c_points, triangles_indexes)

        # Compute area
        mean_area_difference1_14 = compute_mean_triangles_area(img_genuine_1_tri_points, img_genuine_14_tri_points)
        mean_area_difference1_5 = compute_mean_triangles_area(img_genuine_1_tri_points, img_genuine_5_tri_points)
        mean_area_difference1_a = compute_mean_triangles_area(img_genuine_1_tri_points, img_beauty_a_tri_points)
        mean_area_difference1_b = compute_mean_triangles_area(img_genuine_1_tri_points, img_beauty_b_tri_points)
        mean_area_difference1_c = compute_mean_triangles_area(img_genuine_1_tri_points, img_beauty_c_tri_points)

        mean_area.extend(
            [mean_area_difference1_14, mean_area_difference1_5, mean_area_difference1_a, mean_area_difference1_b,
             mean_area_difference1_c])

        # Compute centroid
        centroid_distances1_14 = compute_mean_centroids_distances(img_genuine_1_tri_points, img_genuine_14_tri_points)
        centroid_distances1_5 = compute_mean_centroids_distances(img_genuine_1_tri_points, img_genuine_5_tri_points)
        centroid_distances1_a = compute_mean_centroids_distances(img_genuine_1_tri_points, img_beauty_a_tri_points)
        centroid_distances1_b = compute_mean_centroids_distances(img_genuine_1_tri_points, img_beauty_b_tri_points)
        centroid_distances1_c = compute_mean_centroids_distances(img_genuine_1_tri_points, img_beauty_c_tri_points)

        centroids.extend(
            [centroid_distances1_14, centroid_distances1_5, centroid_distances1_a, centroid_distances1_b,
             centroid_distances1_c])

        # Compute cosine similarity
        angles_distances1_14 = compute_mean_angles_distances(img_genuine_1_tri_points, img_genuine_14_tri_points)
        angles_distances1_5 = compute_mean_angles_distances(img_genuine_1_tri_points, img_genuine_5_tri_points)
        angles_distances1_a = compute_mean_angles_distances(img_genuine_1_tri_points, img_beauty_a_tri_points)
        angles_distances1_b = compute_mean_angles_distances(img_genuine_1_tri_points, img_beauty_b_tri_points)
        angles_distances1_c = compute_mean_angles_distances(img_genuine_1_tri_points, img_beauty_c_tri_points)

        angles.extend(
            [angles_distances1_14, angles_distances1_5, angles_distances1_a, angles_distances1_b,
             angles_distances1_c])

        labels.extend([0, 0, 1, 1, 1])

    mean_area = np.array(mean_area)
    mean_area_labels = np.array(mean_area_labels)
    centroids = np.array(centroids)
    centroids_labels = np.array(centroids_labels)
    angles = np.array(angles)
    angles_labels = np.array(angles_labels)
    # Normalize
    mean_area = normalize([mean_area], norm='max').flatten()
    centroids = normalize([centroids], norm='max').flatten()
    angles = normalize([angles], norm='max').flatten()

    dataset = np.column_stack([mean_area, centroids, angles])

    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=23)

    ####### SVM
    svm = svm.SVC(C=1, kernel='linear')
    svm.fit(x_train, y_train)
    svm_predicted = svm.predict(x_test)
    print('Accuracy score svm: ', accuracy_score(y_test, svm_predicted))
