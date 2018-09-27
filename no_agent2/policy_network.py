import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization, Embedding, Input
from keras.optimizers import sgd
from keras import callbacks
from keras.preprocessing import sequence

class PolicyNetwork:
    def __init__(self, input_dim, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        # LSTM 신경망
        self.model = Sequential()

        self.model.add(LSTM(256, input_shape=(5, 15),
                            return_sequences=True, stateful=False, dropout=0.5))

        # 기존 LSTM 모델
        # self.model.add(LSTM(256, input_shape=input_dim,
        #                     return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(3))
        # self.model.add(Dense(units=3, activation='softmax'))
        self.model.add(Activation('linear'))


        self.model.compile(optimizer=sgd(lr=lr), loss='mse', metrics=['accuracy'])
        self.prob = None

    def predict(self, x):
        return self.model.predict(x)[0]

    def fit(self, x_train, y_train, x_test, y_test, epochs=1000, batch_size=10, model_path=None):
        tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        model_checkpoint = callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        self.model.fit(x_train, y_train,
                       batch_size=batch_size, epochs=epochs,
                       validation_data=(x_test, y_test),
                       callbacks=[tensorboard, model_checkpoint, early_stopping])


    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)