from image_alterations_detector.classifiers.mlp_svm_rf import MlpSvmRf
from image_alterations_detector.dataset.altered_dataset.altered_dataset_utils import load_altered_dataset
from image_alterations_detector.dataset.altered_dataset.altered_descriptors import compute_two_image_descriptors

if __name__ == '__main__':
    dataset_path = '/Users/luca/Desktop/altered'
    genuine, beauty = load_altered_dataset(dataset_path)
    image_index = 3
    descriptor1_14_mean_area, descriptor1_14_matrix, descriptor1_14_lbp = compute_two_image_descriptors(
        genuine[0][image_index],
        genuine[2][image_index])
    descriptor1_c_mean_area, descriptor1_c_matrix, descriptor1_c_lbp = compute_two_image_descriptors(
        genuine[0][image_index],
        beauty[2][image_index])
    print(descriptor1_14_matrix.shape, descriptor1_14_lbp.shape)
    print(descriptor1_c_matrix.shape, descriptor1_c_lbp.shape)

    multi_clf_matrices = MlpSvmRf('affine matrices', svm_c=100, svm_kernel='linear', rf_max_depth=5)
    multi_clf_matrices.load_models('matrices_descriptors')
    multi_clf_lbp = MlpSvmRf('lbp', svm_c=1000, svm_kernel='linear', rf_max_depth=13)
    multi_clf_lbp.load_models('lbp_descriptors')
