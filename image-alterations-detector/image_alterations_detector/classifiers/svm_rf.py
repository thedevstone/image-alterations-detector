import os

import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from image_alterations_detector.file_system.path_utilities import get_folder_path_from_root, get_model_path


class SvmRf:
    def __init__(self):
        self.svm = SVC(C=1, kernel='linear')
        self.rf = RandomForestClassifier(max_depth=7, random_state=0)  # 7
        self.svm_rf = VotingClassifier(estimators=[('svm', self.svm), ('rf', self.rf)],
                                       voting='hard', weights=[1, 1], flatten_transform=True, n_jobs=-1)

    def fit(self, x_train, y_train):
        self.svm_rf.fit(x_train, y_train)

    def evaluate(self, x_test, y_test):
        predicted = self.svm_rf.predict(x_test)
        print("SVM/RF accuracy score:", accuracy_score(y_test, predicted))

    def predict(self, x_test):
        predicted = self.svm_rf.predict(x_test)
        return predicted

    def save(self, file_name):
        joblib.dump(self.svm_rf, os.path.join(get_folder_path_from_root('models'), file_name))

    def load(self, file_name):
        self.svm_rf = joblib.load(get_model_path(file_name))
