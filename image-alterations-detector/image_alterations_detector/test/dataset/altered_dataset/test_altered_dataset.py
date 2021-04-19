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
    multi_clf_matrices = MlpSvmRf('affine_matrices')
    multi_clf_matrices.load_models('affine_matrices')
    # Load lbp model
    multi_clf_lbp = MlpSvmRf('lbp')
    multi_clf_lbp.load_models('lbp')

    matrix_result_1_14 = multi_clf_matrices.predict_one(descriptor1_14_matrix)
    lbp_result_1_14 = multi_clf_lbp.predict_one(descriptor1_14_lbp)
    print("1 - 14 result:", matrix_result_1_14, lbp_result_1_14)
    matrix_result_1_c = multi_clf_matrices.predict_one(descriptor1_c_matrix)
    lbp_result_1_c = multi_clf_lbp.predict_one(descriptor1_c_lbp)
    print("1 - c result:", matrix_result_1_c, lbp_result_1_c)
