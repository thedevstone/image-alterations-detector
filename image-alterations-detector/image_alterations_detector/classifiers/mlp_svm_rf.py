from image_alterations_detector.classifiers.mlp import Mlp
from image_alterations_detector.classifiers.svm_rf import SvmRf


class MlpSvmRf:
    def __init__(self, feature_name, epochs=500, batch_size=64):
        self.feature_name = feature_name
        self.svm_rf = SvmRf(self.feature_name)
        self.mlp = Mlp(self.feature_name, epochs=epochs, batch_size=batch_size)

    def create_model(self, svm_c, svm_gamma, svm_kernel, rf_max_depth, input_shape_length, layer1, layer2, activation,
                     dropout):
        self.mlp.create_model(input_shape_length, layer1=layer1, layer2=layer2, activation=activation, dropout=dropout)
        self.svm_rf.create_model(svm_c, svm_gamma, svm_kernel, rf_max_depth)

    def fit(self, x_train, y_train, class_weight, grid_search):
        self.svm_rf.fit(x_train, y_train, grid_search)
        self.mlp.fit(x_train, y_train, class_weight)

    def evaluate(self, x_test, y_test):
        self.svm_rf.evaluate(x_test, y_test)
        self.mlp.evaluate(x_test, y_test)

    def predict_one(self, x_test):
        svm_rf_predicted = 1 - self.svm_rf.predict(x_test)[0][1]
        mlp_predicted = 1 - self.mlp.predict(x_test)[0][0]
        return self.feature_name, round(svm_rf_predicted, 2), round(mlp_predicted, 2)

    def save_models(self):
        self.svm_rf.save('{}-svm_rf.pkl'.format(self.feature_name))
        self.mlp.save_model('{}-mlp.h5'.format(self.feature_name))

    def load_models(self, file_prefix):
        self.svm_rf.load('{}-svm_rf.pkl'.format(file_prefix))
        self.mlp.load_model('{}-mlp.h5'.format(file_prefix))
