import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

from image_alterations_detector.classifiers.mlp_svm_rf import MlpSvmRf
from image_alterations_detector.dataset.altered_dataset.altered_descriptors import compute_altered_descriptors_beauty, \
    AFFINE_MATRICES_DIM, LBP_DIM, ANGLES_DIM, compute_altered_descriptors_distortion, CENTROIDS_DIM, MEAN_AREAS_DIM
from image_alterations_detector.file_system.path_utilities import get_folder_path_from_root


def train_altered_descriptors_beauty():
    dataset_path = '/Users/luca/Desktop/altered'
    lbp_descriptors, labels = compute_altered_descriptors_beauty(dataset_path, 10)
    # Class weights
    class_weight = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print("Class weights:", class_weight)

    print("LBP shape:", lbp_descriptors.shape)
    x_train_lbp_descriptors, x_test_lbp_descriptors, y_train_lbp_descriptors, y_test_lbp_descriptors = \
        train_test_split(lbp_descriptors, labels, test_size=0.2, random_state=23)
    # Min max
    lbp_scaler = StandardScaler()
    x_train_lbp_descriptors = lbp_scaler.fit_transform(x_train_lbp_descriptors)
    x_test_lbp_descriptors = lbp_scaler.transform(x_test_lbp_descriptors)
    # Train on lbp
    multi_clf_lbp = MlpSvmRf('lbp')
    multi_clf_lbp.create_model(rf_max_depth=13, svm_c=100, svm_gamma=0.001, svm_kernel='rbf',
                               input_shape_length=LBP_DIM, layer1=50, layer2=10, activation='tanh', dropout=0.2)
    multi_clf_lbp.fit(x_train_lbp_descriptors, y_train_lbp_descriptors,
                      class_weight={0: class_weight[0], 1: class_weight[1]}, grid_search=False)
    # Evaluate and save
    multi_clf_lbp.evaluate(x_test_lbp_descriptors, y_test_lbp_descriptors)
    multi_clf_lbp.save_models()
    joblib.dump(lbp_scaler, os.path.join(get_folder_path_from_root('models'), 'lbp_scaler.pkl'))


def train_altered_descriptors_distortion():
    dataset_path = '/Users/luca/Desktop/altered'
    angles_descriptors, mean_area_descriptors, centroid_descriptors, matrices_descriptors, labels = \
        compute_altered_descriptors_distortion(dataset_path, 5)

    # Class weights
    class_weight = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print("Class weights:", class_weight)

    # ------------------------------------------------ TRAIN ON ANGLES ---------------------------------------
    print("Angles shape:", angles_descriptors.shape)
    x_train_angles_descriptors, x_test_angles_descriptors, y_train_angles_descriptors, y_test_angles_descriptors \
        = train_test_split(angles_descriptors, labels, test_size=0.2, random_state=23)
    # Min max
    angles_scaler = StandardScaler()
    x_train_angles_descriptors = angles_scaler.fit_transform(x_train_angles_descriptors)
    x_test_angles_descriptors = angles_scaler.transform(x_test_angles_descriptors)
    print("Training on angles")
    multi_clf_angles = MlpSvmRf('angles')
    multi_clf_angles.create_model(rf_max_depth=7, svm_c=10, svm_gamma=0.0001, svm_kernel='rbf',
                                  input_shape_length=ANGLES_DIM, layer1=200, layer2=50, activation='tanh',
                                  dropout=0.2)
    multi_clf_angles.fit(x_train_angles_descriptors, y_train_angles_descriptors,
                         class_weight={0: class_weight[0], 1: class_weight[1]}, grid_search=False)
    # Evaluate and save
    multi_clf_angles.evaluate(x_test_angles_descriptors, y_test_angles_descriptors)
    multi_clf_angles.save_models()

    # ------------------------------------------------ TRAIN ON AREAS ---------------------------------------
    print("Areas shape:", mean_area_descriptors.shape)
    x_train_areas_descriptors, x_test_areas_descriptors, y_train_areas_descriptors, y_test_areas_descriptors \
        = train_test_split(mean_area_descriptors, labels, test_size=0.2, random_state=23)
    # Min max
    areas_scaler = StandardScaler()
    x_train_areas_descriptors = areas_scaler.fit_transform(x_train_areas_descriptors)
    x_test_areas_descriptors = areas_scaler.transform(x_test_areas_descriptors)
    print("Training on areas")
    multi_clf_areas = MlpSvmRf('areas')
    multi_clf_areas.create_model(rf_max_depth=7, svm_c=10, svm_gamma=0.0001, svm_kernel='rbf',
                                 input_shape_length=MEAN_AREAS_DIM, layer1=200, layer2=50, activation='tanh',
                                 dropout=0.2)
    multi_clf_areas.fit(x_train_areas_descriptors, y_train_areas_descriptors,
                        class_weight={0: class_weight[0], 1: class_weight[1]}, grid_search=False)
    # Evaluate and save
    multi_clf_areas.evaluate(x_test_areas_descriptors, y_test_areas_descriptors)
    multi_clf_areas.save_models()

    # ------------------------------------------------ TRAIN ON CENTROIDS ---------------------------------------
    print("Centroids shape:", centroid_descriptors.shape)
    x_train_centroids_descriptors, x_test_centroids_descriptors, y_train_centroids_descriptors, y_test_centroids_descriptors \
        = train_test_split(centroid_descriptors, labels, test_size=0.2, random_state=23)
    # Min max
    centroids_scaler = StandardScaler()
    x_train_centroids_descriptors = centroids_scaler.fit_transform(x_train_centroids_descriptors)
    x_test_centroids_descriptors = centroids_scaler.transform(x_test_centroids_descriptors)
    print("Training on centroids")
    multi_clf_centroids = MlpSvmRf('centroids')
    multi_clf_centroids.create_model(rf_max_depth=7, svm_c=10, svm_gamma=0.0001, svm_kernel='rbf',
                                     input_shape_length=CENTROIDS_DIM, layer1=200, layer2=50, activation='tanh',
                                     dropout=0.2)
    multi_clf_centroids.fit(x_train_centroids_descriptors, y_train_centroids_descriptors,
                            class_weight={0: class_weight[0], 1: class_weight[1]}, grid_search=False)
    # Evaluate and save
    multi_clf_centroids.evaluate(x_test_centroids_descriptors, y_test_centroids_descriptors)
    multi_clf_centroids.save_models()

    # ------------------------------------------------ TRAIN ON MATRICES ---------------------------------------
    print("Matrices shape:", matrices_descriptors.shape)
    x_train_matrices_descriptors, x_test_matrices_descriptors, y_train_matrices_descriptors, y_test_matrices_descriptors \
        = train_test_split(matrices_descriptors, labels, test_size=0.2, random_state=23)
    # Min max
    matrices_scaler = StandardScaler()
    x_train_matrices_descriptors = matrices_scaler.fit_transform(x_train_matrices_descriptors)
    x_test_matrices_descriptors = matrices_scaler.transform(x_test_matrices_descriptors)
    print("Training on affine matrices")
    multi_clf_matrices = MlpSvmRf('affine_matrices')
    multi_clf_matrices.create_model(rf_max_depth=7, svm_c=10, svm_gamma=0.0001, svm_kernel='rbf',
                                    input_shape_length=AFFINE_MATRICES_DIM, layer1=300, layer2=50, activation='tanh',
                                    dropout=0.2)
    multi_clf_matrices.fit(x_train_matrices_descriptors, y_train_matrices_descriptors,
                           class_weight={0: class_weight[0], 1: class_weight[1]}, grid_search=False)
    # Evaluate and save
    multi_clf_matrices.evaluate(x_test_matrices_descriptors, y_test_matrices_descriptors)
    multi_clf_matrices.save_models()

    # Save scalers
    joblib.dump(angles_scaler, os.path.join(get_folder_path_from_root('models'), 'angles_scaler.pkl'))
    joblib.dump(matrices_scaler, os.path.join(get_folder_path_from_root('models'), 'matrices_scaler.pkl'))


if __name__ == '__main__':
    train_altered_descriptors_distortion()
    # train_altered_descriptors_beauty()
