import os

from keras import models
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization

from image_alterations_detector.file_system.path_utilities import get_folder_path_from_root, get_model_path


class Mlp:
    def __init__(self):
        self.model = None
        self.input_shape = None

    def create_model(self, input_shape_length, layer1=300, layer2=50, layer3=None):
        """ Init mlp

        :param input_shape_length: the dimension of the input
        :param layer1: first layer neurons. default to 300
        :param layer2: second layer neurons. default to 50 (try 100)
        :param layer3: third layer neurons. default to None (try 10)
        """
        self.input_shape = (input_shape_length,)
        # Create the model
        self.model = models.Sequential()
        self.model.add(Dense(layer1, input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(layer2, activation='tanh'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        if layer3:
            self.model.add(Dense(layer3, activation='tanh'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, epochs=500, batch_size=64):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

    def evaluate(self, x_test, y_test):
        test_results = self.model.evaluate(x_test, y_test, verbose=1)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}')

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save_model(self, file_name):
        self.model.save(os.path.join(get_folder_path_from_root('models'), file_name))

    def load_model(self, file_name):
        self.model = models.load_model(get_model_path(file_name))
