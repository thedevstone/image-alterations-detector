import numpy as np
from sklearn.model_selection import train_test_split

from image_alterations_detector.classifiers.mlp_svm_rf import MlpSvmRf
from image_alterations_detector.dataset.altered_dataset.altered_features import compute_altered_descriptors


def train_altered_descriptors():
    dataset_path = '/Users/luca/Desktop/altered'
    mean_area_descriptors, matrices_descriptors, lbp_descriptors, labels = compute_altered_descriptors(dataset_path)

    descriptors: np.ndarray = np.column_stack([matrices_descriptors, lbp_descriptors])
    print("Big descriptor shape:", descriptors.shape)
    x_train_descriptors, x_test_descriptors, y_train_descriptors, y_test_descriptors = train_test_split(descriptors,
                                                                                                        labels,
                                                                                                        test_size=0.2,
                                                                                                        random_state=23)
    # Train
    multi_clf_matrices = MlpSvmRf()
    multi_clf_matrices.create_model(x_test_descriptors.shape[1])
    multi_clf_matrices.fit(x_train_descriptors, y_train_descriptors)
    multi_clf_matrices.evaluate(x_test_descriptors, y_test_descriptors)
    multi_clf_matrices.save_models('total_descriptors')
    to_predict = np.expand_dims(x_test_descriptors[0], 0)

    # Test on two images
    result = multi_clf_matrices.predict_one(to_predict)
    print(result)


if __name__ == '__main__':
    train_altered_descriptors()
