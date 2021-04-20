import cv2
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

from image_alterations_detector.classifiers.mlp_svm_rf import MlpSvmRf
from image_alterations_detector.dataset.altered_dataset.altered_descriptors import compute_two_image_descriptors
from image_alterations_detector.file_system.path_utilities import get_image_path, get_model_path


def test_two_images(img1, img2):
    descriptor1_2_angles, descriptor1_2_area, descriptor1_2_matrix, descriptor1_2_lbp = compute_two_image_descriptors(
        img1, img2)

    # Load angles model
    angles_scaler: RobustScaler = joblib.load(get_model_path('angles_scaler.pkl'))
    descriptor1_2_angles = angles_scaler.transform(np.expand_dims(descriptor1_2_angles, 0))
    multi_clf_angles = MlpSvmRf('angles')
    multi_clf_angles.load_models('angles')
    # Load affine matrices model
    matrices_scaler: RobustScaler = joblib.load(get_model_path('matrices_scaler.pkl'))
    descriptor1_2_matrix = matrices_scaler.transform(np.expand_dims(descriptor1_2_matrix, 0))
    multi_clf_matrices = MlpSvmRf('affine_matrices')
    multi_clf_matrices.load_models('affine_matrices')
    # Load lbp model
    lbp_scaler: RobustScaler = joblib.load(get_model_path('lbp_scaler.pkl'))
    descriptor1_2_lbp = lbp_scaler.transform(np.expand_dims(descriptor1_2_lbp, 0))
    multi_clf_lbp = MlpSvmRf('lbp')
    multi_clf_lbp.load_models('lbp')

    angles_result_1_2 = multi_clf_angles.predict_one(descriptor1_2_angles)
    matrix_result_1_2 = multi_clf_matrices.predict_one(descriptor1_2_matrix)
    lbp_result_1_2 = multi_clf_lbp.predict_one(descriptor1_2_lbp)
    return angles_result_1_2, matrix_result_1_2, lbp_result_1_2


def test_on_extra_dataset_images(image1_name, image2_name):
    img1 = cv2.imread(get_image_path(image1_name))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(get_image_path(image2_name))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return test_two_images(img1, img2)
