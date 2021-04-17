import numpy as np
from keras import models
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization

if __name__ == '__main__':
    # Multi-classifier
    svm_mean_area = SVC(C=1, kernel='linear')
    rf_mean_area = RandomForestClassifier(max_depth=7, random_state=0)  # 7

    multi_mean_area = VotingClassifier(estimators=[
        ('svm', svm_mean_area), ('rf', rf_mean_area)],
        voting='hard', weights=[1, 1],
        flatten_transform=True, n_jobs=-1)
    multi_mean_area.fit(x_train_mean_area, y_train_mean_area)
    predicted_mean_area = multi_mean_area.predict(x_test_mean_area)
    print("SVM/RF mean area accuracy score:", accuracy_score(y_test_mean_area, predicted_mean_area))

    svm_matrices = SVC(C=1, kernel='linear')
    rf_matrices = RandomForestClassifier(max_depth=7, random_state=0)  # 7

    multi_matrices = VotingClassifier(estimators=[
        ('svm', svm_matrices), ('rf', rf_matrices)],
        voting='hard', weights=[1, 1],
        flatten_transform=True, n_jobs=-1)
    multi_matrices.fit(x_train_matrices, y_train_matrices)
    predicted_matrices = multi_matrices.predict(x_test_matrices)
    print("SVM/RF matrices accuracy score:", accuracy_score(y_test_matrices, predicted_matrices))

    print("\nFINAL EVALUATION")
    mean_area_matrices_predicted = np.column_stack((predicted_mean_area, predicted_matrices))
    mean_area_matrices_predicted = stats.mode(mean_area_matrices_predicted, axis=1)[0]
    print("\nMulti classifier accuracy score:", accuracy_score(y_test_mean_area, mean_area_matrices_predicted))

    print("Testing keras")
    # Configuration options
    feature_vector_length = 678
    # y_train = to_categorical(y_train_matrices, 1)
    # y_test = to_categorical(y_test_matrices, 1)
    # Set the input shape
    input_shape = (feature_vector_length,)
    print(f'Feature shape: {input_shape}')
    # Create the model
    model = models.Sequential()
    model.add(Dense(350, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(100, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))
    # Configure the model and start training
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_matrices, y_train_matrices, epochs=200, batch_size=128, verbose=1, validation_split=0.2)
    # Test the model after training
    test_results = model.evaluate(x_test_matrices, y_test_matrices, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}')
