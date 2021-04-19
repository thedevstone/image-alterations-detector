import cv2

from image_alterations_detector.classifiers.mlp_svm_rf import MlpSvmRf
from image_alterations_detector.dataset.altered_dataset.altered_descriptors import compute_two_image_descriptors
from image_alterations_detector.file_system.path_utilities import get_image_path

if __name__ == '__main__':
    print('Testing on images')
    img1 = cv2.imread(get_image_path('img1.jpg'))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(get_image_path('img2.jpg'))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    descriptor1_2_area, descriptor1_2_matrix, descriptor1_2_lbp = compute_two_image_descriptors(img1, img2)
    print(descriptor1_2_matrix.shape, descriptor1_2_lbp.shape)

    # Load affine matrices model
    multi_clf_matrices = MlpSvmRf('affine_matrices', svm_c=None, svm_kernel=None, rf_max_depth=None)
    multi_clf_matrices.create_model(descriptor1_2_matrix.shape[1], layer1=100, layer2=150, activation='tanh',
                                    dropout=0.5)
    multi_clf_matrices.load_models('affine_matrices')
    # Load lbp model
    multi_clf_lbp = MlpSvmRf('lbp', svm_c=1000, svm_kernel='linear', rf_max_depth=13)
    multi_clf_lbp.create_model(descriptor1_2_lbp.shape[1], layer1=300, layer2=100, activation='tanh', dropout=0.5)
    multi_clf_lbp.load_models('lbp')

    matrix_result = multi_clf_matrices.predict_one(descriptor1_2_matrix)
    matrix_result_beauty = multi_clf_lbp.predict_one(descriptor1_2_lbp)
    print(matrix_result)
    print(matrix_result_beauty)
