import pandas as pd
import numpy as np

class stock_data():

    def __init__(self, df):
        self.data = df.set_index(['timestamp']).query('volume!=0').copy()
        self.proc_data = self.data.copy()

    def next_n_return(self, n):
        for col in self.data.columns:
            self.proc_data[f'{col}_next_{n}_day_return'] = \
            (self.proc_data[col] - self.proc_data[col].shift(n))/self.proc_data[col].shift(n)

    def create_trend_label(self, period):
        self.proc_data[f'next_{period}_return_label'] = (self.proc_data.close - \
            self.proc_data.close.shift(period)).apply(lambda x: 1 if x>0 else -1)
        self.tar = f'next_{period}_return_label'

    def rolling_average(self, windows):
        for col in self.data.columns:
            self.proc_data[f'{col}_{windows}_rolling_mean'] = self.data[col].rolling(windows).mean()

    def rolling_std(self, windows):
        for col in self.data.columns:
            self.proc_data[f'{col}_{windows}_rolling_std'] = self.data[col].rolling(windows).std()

    def preprocess(self, ns=[None], windows=[None]):
        try:
            for n in ns:
                self.next_n_return(n)
        except:
            print('no ns')

        try:
            for w in windows:
                self.rolling_average(w)
                self.rolling_std(w)
        except:
            print('no windows')

        self.skip = max(max(ns), max(windows))

    def create_train_test(self, trn_len, classifier=False):
        tar = self.tar if classifier else 'close'
        self.data_xtrain, self.data_xtest = self.proc_data.iloc[self.skip:trn_len-1], self.proc_data.iloc[trn_len-1:-1]
        self.data_ytrain, self.data_ytest = self.proc_data[tar][self.skip+1:trn_len], self.proc_data[tar][trn_len:]
