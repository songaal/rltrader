import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization, Dropout, Flatten
from keras.optimizers import sgd
from keras import callbacks

class PolicyNetwork:
    def __init__(self, input_dim, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        # LSTM 신경망
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(1, input_dim)))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Flatten())
        self.model.add(Activation('sigmoid'))

        self.model.compile(optimizer=sgd(lr=lr), loss='mse')
        self.prob = None

    def reset(self):
        self.prob = None

    def predict(self, x):
        # self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        self.prob = self.model.predict(x)[0]
        return self.prob

    def fit(self, x_train, y_train, x_test, y_test, epochs=1000, batch_size=10):
        tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[tensorboard])

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)