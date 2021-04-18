import os

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

from image_alterations_detector.file_system.path_utilities import get_folder_path_from_root, get_model_path


class SvmRf:
    def __init__(self, feature_name):
        """ Create a multi-classifier with SVM and Random Forest

        """
        self.feature_name = feature_name
        self.svm = SVC(C=1, kernel='linear')
        self.rf = RandomForestClassifier(max_depth=7, random_state=0)  # 7
        self.svm_rf = VotingClassifier(estimators=[('svm', self.svm), ('rf', self.rf)],
                                       voting='hard', weights=[1, 1], flatten_transform=True, n_jobs=-1)

    def fit(self, x_train, y_train):
        print('Training SVM_RF on', self.feature_name)
        self.svm_rf.fit(x_train, y_train)
        y_pred = self.svm_rf.predict(x_train)
        print('Train SVM_RF accuracy:', accuracy_score(y_train, y_pred))

    def evaluate(self, x_test, y_test):
        y_pred = self.svm_rf.predict(x_test)
        print('Evaluating SVM_RF performance on', self.feature_name)
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Precision:', precision_score(y_test, y_pred), 'Recall:', recall_score(y_test, y_pred))
        print('Confusion matrix:')
        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.show()

    def predict(self, x_test):
        predicted = self.svm_rf.predict(x_test)
        return predicted

    def save(self, file_name):
        joblib.dump(self.svm_rf, os.path.join(get_folder_path_from_root('models'), file_name))

    def load(self, file_name):
        self.svm_rf = joblib.load(get_model_path(file_name))
