import numpy as np

from image_alterations_detector.classifiers.mlp import Mlp
from image_alterations_detector.classifiers.svm_rf import SvmRf


class MlpSvmRf:
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.svm_rf = SvmRf(self.feature_name)
        self.mlp = Mlp(self.feature_name)

    def create_model(self, input_shape_length, layer1=300, layer2=50):
        self.mlp.create_model(input_shape_length, layer1, layer2)

    def fit(self, x_train, y_train, epochs=500, batch_size=32):
        self.mlp.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        self.svm_rf.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        self.svm_rf.evaluate(x_test, y_test)
        self.mlp.evaluate(x_test, y_test)

    def predict_one(self, x_test):
        svm_rf_predicted = self.svm_rf.predict(x_test)
        mlp_predicted = self.mlp.predict(x_test)
        return np.array([svm_rf_predicted[0], mlp_predicted[0]])

    def save_models(self, file_prefix):
        self.svm_rf.save('{}-svm_rf.pkl'.format(file_prefix))
        self.mlp.save_model('{}-mlp.h5'.format(file_prefix))

    def load_models(self, file_prefix):
        self.svm_rf.load('{}-svm_rf.pkl'.format(file_prefix))
        self.mlp.load_model('mlp.h5'.format(file_prefix))
