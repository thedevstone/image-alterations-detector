import os

from keras import models
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from image_alterations_detector.file_system.path_utilities import get_folder_path_from_root, get_model_path


class Mlp:
    def __init__(self, feature_name, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.keras_clf = None
        self.input_shape = None
        self.feature_name = feature_name

    def model_builder(self, input_shape_length, layer1, layer2, activation, dropout):
        """ Initialize mlp

        :param input_shape_length: the dimension of the input
        :param layer1: first layer neurons. default to 300
        :param layer2: second layer neurons. default to 50 (try 100)
        :param activation: activation function
        :param dropout: the dropout rate
        """
        self.input_shape = (input_shape_length,)
        # Create the model
        model = models.Sequential()
        model.add(Dense(layer1, input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(layer2, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def create_model(self, input_shape_length, layer1, layer2, activation, dropout):
        self.keras_clf = KerasClassifier(self.model_builder, input_shape_length=input_shape_length, layer1=layer1,
                                         layer2=layer2, activation=activation, dropout=dropout)

    def fit(self, x_train, y_train, grid_search, class_weight):
        print('Training MLP on', self.feature_name)
        if grid_search:
            activation = ['tanh', 'relu']
            dropout = [0.2, 0.5]
            layer1 = [100, 300, 500]
            layer2 = [10, 50, 100]
            param_grid = dict(layer1=layer1, layer2=layer2, activation=activation, dropout=dropout)
            grid = GridSearchCV(estimator=self.keras_clf, param_grid=param_grid, cv=5, return_train_score=False)
            grid_result = grid.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0,
                                   class_weight=class_weight)
            # Summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            self.keras_clf = grid.best_estimator_
        else:
            # Normal fit
            self.keras_clf.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            y_pred = self.keras_clf.predict(x_train)
            print('MLP accuracy on train for{}:'.format(self.feature_name), accuracy_score(y_train, y_pred))

    def evaluate(self, x_test, y_test):
        y_pred = self.keras_clf.predict(x_test)
        print('MLP performance on test for', self.feature_name)
        print('Accuracy:', accuracy_score(y_test, y_pred), 'Precision:', precision_score(y_test, y_pred), 'Recall:',
              recall_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.axis('off')
        plt.show()

    def predict(self, x_test):
        return self.keras_clf.predict(x_test)

    def save_model(self, file_name):
        self.keras_clf.model.save(os.path.join(get_folder_path_from_root('models'), file_name))

    def load_model(self, file_name):
        model = models.load_model(get_model_path(file_name))
        self.keras_clf = KerasClassifier(model)
