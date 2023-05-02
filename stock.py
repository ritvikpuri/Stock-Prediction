import numpy as np
import matplotlib as plt
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
# from keras.models import Sequential
# import keras.layers as layers

# loading dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# since we are only using open stock prices
open_values = dataset_train.iloc[:, 1:2].values

sc = sk.preprocessing.MinMaxScaler(feature_range=(0, 1))
open_values_scaled = sc.fit_transform(open_values)

X_train, Y_train = [], []
steps = 10
for i in range(steps, 1258):
    X_train.append(open_values_scaled[i-steps, 0])
    Y_train.append(open_values_scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# rnn

regressor = Sequential()

regressor.add(layers.LSTM(units = 50, return_sequences=True, input_shape= (X_train.shape[1], 1)))
regressor.add(layers.Dropout(0.2))

regressor.add(layers.LSTM(units = 50, return_sequences=True))
regressor.add(layers.Dropout(0.2))
regressor.add(layers.LSTM(units = 50, return_sequences=True))
regressor.add(layers.Dropout(0.2))
regressor.add(layers.LSTM(units = 50))
regressor.add(layers.Dropout(0.2))

regressor.add(layers.Dense(units=1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

#visualising

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
