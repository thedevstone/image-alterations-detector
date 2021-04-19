import cv2

from image_alterations_detector.classifiers.mlp_svm_rf import MlpSvmRf
from image_alterations_detector.dataset.altered_dataset.altered_descriptors import compute_two_image_descriptors
from image_alterations_detector.file_system.path_utilities import get_image_path

if __name__ == '__main__':
    print('Testing on images')
    img1 = cv2.imread(get_image_path('test_luca2.jpg'))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(get_image_path('test_luca1_beauty.jpeg'))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3 = cv2.imread(get_image_path('test_luca1_liquify.png'))
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    descriptor1_2_area, descriptor1_2_matrix, descriptor1_2_lbp = compute_two_image_descriptors(img1, img2)
    descriptor1_3_area, descriptor1_3_matrix, descriptor1_3_lbp = compute_two_image_descriptors(img1, img3)
    # Load affine matrices model
    multi_clf_matrices = MlpSvmRf('affine_matrices')
    multi_clf_matrices.load_models('affine_matrices')
    # Load lbp model
    multi_clf_lbp = MlpSvmRf('lbp')
    multi_clf_lbp.load_models('lbp')

    matrix_result_1_2 = multi_clf_matrices.predict_one(descriptor1_2_matrix)
    lbp_result_1_2 = multi_clf_lbp.predict_one(descriptor1_2_lbp)
    print("1 - 2 result:", matrix_result_1_2, lbp_result_1_2)
    matrix_result_1_3 = multi_clf_matrices.predict_one(descriptor1_3_matrix)
    lbp_result_1_3 = multi_clf_lbp.predict_one(descriptor1_3_lbp)
    print("1 - 3 result:", matrix_result_1_3, lbp_result_1_3)
