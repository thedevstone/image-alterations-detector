import os

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from image_alterations_detector.file_system.path_utilities import get_folder_path_from_root, get_model_path


class SvmRf:
    def __init__(self, feature_name):
        """ Create a multi-classifier with SVM and Random Forest

        """
        self.feature_name = feature_name
        self.svm = SVC(probability=True)  # C=1, kernel='linear'
        self.rf = RandomForestClassifier(random_state=0)  # max_depth=7
        self.svm_rf = VotingClassifier(estimators=[('svm', self.svm), ('rf', self.rf)],
                                       voting='soft', weights=[1, 1], flatten_transform=True, n_jobs=-1)

    def fit(self, x_train, y_train):
        print('Training SVM_RF on', self.feature_name)
        # GRID SEARCH
        # Use the key for the classifier followed by __ and the attribute
        param_grid = [{'svm__kernel': ['rbf'], 'svm__gamma': [1e-3, 1e-4], 'svm__C': [1, 10, 100, 1000],
                       'rf__max_depth': [5, 7, 9, 11, 13]},
                      {'svm__kernel': ['linear'], 'svm__C': [1, 10, 100, 1000], 'rf__max_depth': [5, 7, 9, 11, 13]}
                      ]
        grid = GridSearchCV(estimator=self.svm_rf, param_grid=param_grid, cv=5, return_train_score=False)
        grid_result = grid.fit(x_train, y_train)
        # Summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        self.svm_rf = grid.best_estimator_

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