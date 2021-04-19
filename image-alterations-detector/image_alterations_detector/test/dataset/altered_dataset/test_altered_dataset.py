from image_alterations_detector.classifiers.mlp_svm_rf import MlpSvmRf
from image_alterations_detector.dataset.altered_dataset.altered_dataset_utils import load_altered_dataset
from image_alterations_detector.dataset.altered_dataset.altered_descriptors import compute_two_image_descriptors

if __name__ == '__main__':
    dataset_path = '/Users/luca/Desktop/altered'
    genuine, beauty = load_altered_dataset(dataset_path)
    image_index = 3
    print('Testing on altered dataset')
    descriptor1_14_mean_area, descriptor1_14_matrix, descriptor1_14_lbp = compute_two_image_descriptors(
        genuine[0][image_index],
        genuine[2][image_index])
    descriptor1_c_mean_area, descriptor1_c_matrix, descriptor1_c_lbp = compute_two_image_descriptors(
        genuine[0][image_index],
        beauty[2][image_index])
    print(descriptor1_14_matrix.shape, descriptor1_14_lbp.shape)
    print(descriptor1_c_matrix.shape, descriptor1_c_lbp.shape)

    # Load affine matrices model
    multi_clf_matrices = MlpSvmRf('affine_matrices', svm_c=1000, svm_kernel='linear', rf_max_depth=5)
    multi_clf_matrices.create_model(descriptor1_14_matrix.shape[1], layer1=100, layer2=50, activation='tanh',
                                    dropout=0.5)
    multi_clf_matrices.load_models('affine_matrices')
    # Load lbp model
    multi_clf_lbp = MlpSvmRf('lbp', svm_c=10000, svm_kernel='linear', rf_max_depth=5)
    multi_clf_lbp.create_model(descriptor1_14_lbp.shape[1], layer1=10, layer2=None, activation='tanh', dropout=0.3)
    multi_clf_lbp.load_models('lbp')

    matrix_result = multi_clf_matrices.predict_one(descriptor1_14_matrix)
    matrix_result_beauty = multi_clf_matrices.predict_one(descriptor1_c_matrix)
    print(matrix_result)
    print(matrix_result_beauty)
