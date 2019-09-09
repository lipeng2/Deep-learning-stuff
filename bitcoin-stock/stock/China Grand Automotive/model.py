import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# keras lib
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LSTM, Input, concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


class lstm_model:

    def __init__(self, data: stock_data, model=None, scaler=None, tar_pos=None):
        self.data = data
        self.model = model
        self.scaler = scaler
        self.tar_pos = tar_pos
        if not model:
            print('Model not created')
        if not scaler:
            print('lstm dataset not created')
        print('Data read: ')
        print(self.data.info())
        print('please create lstm dataset using create_lstm_data()')

    def create_lstm_data(self, input_vars, seq, fs, tar, test_split=0.2):
        self.seq= seq
        self.fs = fs
        self.in_dim = len(input_vars)
        if not self.scaler:
            print('creating new lstm dataset')
        df = self.data[input_vars].copy()
        scaler = MinMaxScaler()
        df[input_vars] = scaler.fit_transform(df[input_vars])
        x,y = [], []
        for i in range(seq, len(df)-seq-fs):
            x.append(df[i-seq:i][input_vars].values)
            y.append(df[i:i+fs][tar].values)

        # assigning class vars
        self.data_x = np.array(x).reshape(-1, seq, len(input_vars))
        self.data_y = np.array(y).reshape(-1, fs)
        self.scaler = scaler
        self.tar_pos = input_vars.index(tar)

        trn_len = int(len(self.data_x)*(1-test_split))
        self.data_xtrain, self.data_xtest = self.data_x[:trn_len], self.data_x[trn_len:]
        self.data_ytrain, self.data_ytest = self.data_y[:trn_len], self.data_y[trn_len:]

    def build_model(self):
        model = Sequential()
        model.add(LSTM(32, input_shape=(self.seq, self.in_dim), return_sequences = True))
        model.add(LSTM(16))
        model.add(Dense(30))
        model.add(Dense(self.fs, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model
        print(model.summary())

    def multi_input_model(self):
        feature_inputs = Intput(shape=(-1, self.n_features), name='feat_inp')
        x = Dense(2*self.n_features)(feature_inputs)
        feature_outputs = Dense(10, activation='linear')(x)

        lstm_inputs = Input(shape=(-1, self.seq, self.in_dim), name='lstm_inp')
        x = LSTM(32, return_sequences=True)(lstm_inputs)
        x = LSTM(16)(x)
        lstm_outputs = Dense(30)(x)

        join = concatenate([feature_outputs, lstm_outputs])
        final_output = Dense(self.fs, activation='linear')(join)
        model = Model([inputs=[feature_inputs, lstm_inputs], outputs=[final_output]])
        self.model = model
        print(model.summary)


    def train(self, bs=1, epochs=100):
        if not self.model:
            print('Model has not been built')
        else:
            es = EarlyStopping(monitor='val_loss', patience=5)
            es2 = EarlyStopping(monitor='loss', patience=10)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, \
                                          patience=3, min_lr=0.0000001, verbose=1)
            self.model.fit(self.data_xtrain, self.data_ytrain, epochs=epochs, batch_size=bs, \
                           validation_split=0.1, shuffle=True, callbacks=[es, reduce_lr])

    def predict(self, x, inverse=False):
        preds = self.model.predict(x)
        return self.inverse(preds) if inverse else preds


    def trend_accuracy(self, x, y):
        preds = self.get_trends(self.predict(x, inverse=True))
        actuals = self.get_trends(self.inverse(y))
        return sum(1 for a,b in zip(preds, actuals) if a==b) / len(preds)

    # helpers

    def inverse(self, vals):
        dmin, dmax = self.scaler.data_min_[self.tar_pos], self.scaler.data_max_[self.tar_pos]
        low, high = self.scaler.feature_range
        inv = np.vectorize(lambda x: (x-low)/(high-low)*(dmax-dmin)+dmin)
        return np.round(inv(vals),4)

    def get_trends(self, x):
        x = x.squeeze()
        return [(1 if x[i+1]-x[i]>0 else -1) for i in range(len(x)-1)]

    def comparison_plot(self, predictions, real):
        plt.figure(figsize=(18,8))
        plt.plot(predictions, label='predictions')
        plt.plot(real, label='real')
        plt.legend(loc='best')
        plt.show()
