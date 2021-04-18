import numpy as np
from sklearn.model_selection import train_test_split

from image_alterations_detector.classifiers.mlp_svm_rf import MlpSvmRf
from image_alterations_detector.dataset.altered_dataset.altered_descriptors import compute_altered_descriptors


def train_altered_descriptors():
    dataset_path = '/Users/luca/Desktop/altered'
    mean_area_descriptors, matrices_descriptors, lbp_descriptors, labels = compute_altered_descriptors(dataset_path)

    # descriptors: np.ndarray = np.column_stack([matrices_descriptors, lbp_descriptors])
    # print("Big descriptor shape:", descriptors.shape)
    print("Matrices shape:", matrices_descriptors.shape)
    x_train_matrices_descriptors, x_test_matrices_descriptors, y_train_matrices_descriptors, y_test_matrices_descriptors = train_test_split(
        matrices_descriptors,
        labels,
        test_size=0.2,
        random_state=23)

    # Train on matrices
    multi_clf_matrices = MlpSvmRf()
    multi_clf_matrices.create_model(x_train_matrices_descriptors.shape[1])
    multi_clf_matrices.fit(x_train_matrices_descriptors, y_train_matrices_descriptors)
    multi_clf_matrices.evaluate(x_test_matrices_descriptors, y_test_matrices_descriptors)
    multi_clf_matrices.save_models('matrices_descriptors')

    print("LBP shape:", lbp_descriptors.shape)
    x_train_lbp_descriptors, x_test_lbp_descriptors, y_train_lbp_descriptors, y_test_lbp_descriptors = train_test_split(
        lbp_descriptors,
        labels,
        test_size=0.2,
        random_state=23)

    # Train on lbp
    multi_clf_lbp = MlpSvmRf()
    multi_clf_lbp.create_model(x_train_lbp_descriptors.shape[1])
    multi_clf_lbp.fit(x_train_lbp_descriptors, y_train_lbp_descriptors)
    multi_clf_lbp.evaluate(x_test_lbp_descriptors, y_test_lbp_descriptors)
    multi_clf_lbp.save_models('lbp_descriptors')

    # Test Matrices
    to_predict_matrices = np.expand_dims(x_test_matrices_descriptors[0], 0)
    result_matrices = multi_clf_matrices.predict_one(to_predict_matrices)
    print(result_matrices)

    # Test Lbp
    to_predict_lbp = np.expand_dims(x_test_lbp_descriptors[0], 0)
    result_lbp = multi_clf_lbp.predict_one(to_predict_lbp)
    print(result_lbp)


if __name__ == '__main__':
    train_altered_descriptors()
