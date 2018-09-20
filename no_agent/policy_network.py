import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import sgd


class PolicyNetwork:
    def __init__(self, input_dim, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        # LSTM 신경망
        self.model = Sequential()

        self.model.add(LSTM(32, input_shape=(10, 1)))
        self.model.add(Dense(1))


        # self.model.add(LSTM(256, input_shape=input_dim, return_sequences=True, stateful=False, dropout=0.5))
        # self.model.add(LSTM(256, input_shape=(input_dim[0], input_dim[1]), return_sequences=True, stateful=False, dropout=0.5))
        # self.model.add(Dense(1))
        # self.model.add(LSTM(256, input_shape=(28, 28), return_sequences=True, stateful=False, dropout=0.5))

        # self.model.add(BatchNormalization())
        # self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        # self.model.add(BatchNormalization())
        # self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))
        # self.model.add(BatchNormalization())
        # self.model.add(Dense(output_dim))
        # self.model.add(Activation('sigmoid'))

        self.model.compile(optimizer=sgd(lr=lr), loss='mse')
        self.prob = None

    def reset(self):
        self.prob = None

    def predict(self, x):
        # self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        self.prob = self.model.predict(x)
        return self.prob

    def fit(self, x_train, y_train, epochs=1000, batch_size=10):
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        print('## training loss and acc ##')
        print(hist.history['loss'])
        print(hist.history['acc'])

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)