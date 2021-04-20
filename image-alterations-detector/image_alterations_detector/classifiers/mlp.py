import os

from keras import models
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, \
    PrecisionRecallDisplay, precision_recall_curve, roc_curve, RocCurveDisplay
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
from image_alterations_detector.file_system.path_utilities import get_folder_path_from_root, get_model_path

tf.get_logger().setLevel('ERROR')


class Mlp:
    def __init__(self, feature_name, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.input_shape = None
        self.feature_name = feature_name

    def create_model(self, input_shape_length, layer1, layer2, activation, dropout):
        self.input_shape = (input_shape_length,)
        # Create the model
        self.model = models.Sequential()
        self.model.add(Dense(layer1, input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout))
        if layer2 is not None:
            self.model.add(Dense(layer2, activation=activation))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(dropout))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, class_weight):
        print('Training MLP on', self.feature_name)
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0,
                       class_weight=class_weight)
        y_pred = self.model.predict(x_train)
        y_pred = [1 * (x[0] >= 0.5) for x in y_pred]
        print('MLP accuracy on train for {}:'.format(self.feature_name), accuracy_score(y_train, y_pred))

    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        y_pred = [1 * (x[0] >= 0.5) for x in y_pred]
        print('MLP performance on test for', self.feature_name)
        print('Accuracy:', accuracy_score(y_test, y_pred), 'Precision:', precision_score(y_test, y_pred), 'Recall:',
              recall_score(y_test, y_pred))
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(cm)
        # Precision recall
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
        # Roc
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        # Figure
        figure: Figure = plt.figure(1, figsize=(15, 6))
        figure.suptitle('MLP on {}'.format(self.feature_name), fontsize=20)
        (ax1, ax2, ax3) = figure.subplots(1, 3)
        ax1.set_title('Confusion matrix')
        cm_display.plot(ax=ax1)
        ax2.set_title('Precision recall')
        pr_display.plot(ax=ax2)
        ax3.set_title('Roc curve')
        roc_display.plot(ax=ax3)
        file_name = '{}-mlp.png'.format(self.feature_name)
        figure.savefig(os.path.join(get_folder_path_from_root('images'), file_name))
        plt.show()

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def save_model(self, file_name):
        self.model.save(os.path.join(get_folder_path_from_root('models'), file_name))

    def load_model(self, file_name):
        self.model = models.load_model(get_model_path(file_name))
