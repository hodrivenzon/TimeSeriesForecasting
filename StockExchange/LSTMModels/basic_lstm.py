import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import yfinance as yf

import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow


class simpleLSTM:
    def __init__(self):
        self.MODEL_PATH = r"models\basic_lstm.h5"


    def create_dataset(self, dataset, look_back=4):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset.iloc[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset.iloc[i + look_back])
            # dataY.append(dataset.iloc[i + look_back][0])
        return np.array(dataX), np.array(dataY)


    def get_features(self, stock_h, num_of_features=1):
        if num_of_features == 1:
            dataset = stock_h[["Close"]]
        elif num_of_features == 2:
            dataset = stock_h[["Close", "Open"]]
        elif num_of_features == 4:
            dataset = stock_h[["Close", "Open", "Low", "High"]]
        elif num_of_features == 5:
            dataset = stock_h[["Close", "Open", "Low", "High", "Volume"]]
        return dataset

    def split_dataset(self, dataset, split_date, initial_data_cut=None, type="start"):
        if initial_data_cut != None:
            split_date_old = pd.Timestamp(initial_data_cut + ' 00:00:00')
            if type == "start":
                dataset = dataset.loc[split_date_old:]
            if type == "end":
                dataset = dataset.loc[:split_date_old]

        split_date = pd.Timestamp(split_date + ' 00:00:00')
        train = dataset.loc[:split_date]
        test = dataset.loc[split_date:]

        # train_size = int(len(dataset) * 0.67)
        # test_size = len(dataset) - train_size
        # train = dataset[0:train_size, :]
        # test = dataset[train_size:len(dataset), :]
        # print(len(train), len(test))
        print(f"Train: {len(train)}, Test: {len(test)}")
        return train, test

    def LSTM_CNN(self, stock_h):
        num_of_features = 4
        dataset = self.get_features(stock_h, num_of_features=num_of_features)
        # train, test = self.split_dataset(dataset, "2020-09-01", initial_data_cut="2020-01-01", type="start")
        # train, test = self.split_dataset(dataset, "2017-02-01")
        # val, test = self.split_dataset(test, "2021-01-01")
        # train, test = self.split_dataset(dataset, "2017-01-01", initial_data_cut="2019-01-01", type="end")
        train, test = self.split_dataset(dataset, "2019-01-01")
        train, val = self.split_dataset(train, "2014-01-01")

        batch_size = 1
        look_back = 3
        EPOCHS = 100

        trainX, trainY = self.create_dataset(train, look_back)
        valX, valY = self.create_dataset(val, look_back)
        testX, testY = self.create_dataset(test, look_back)

        trainX = np.reshape(trainX, (trainX.shape[0], num_of_features, trainX.shape[1]))
        valX = np.reshape(valX, (valX.shape[0], num_of_features, valX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], num_of_features, testX.shape[1]))

        early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

        SAVE = False
        # It can be used to reconstruct the model identically.
        if os.path.exists(self.MODEL_PATH) and SAVE:
            model = tensorflow.keras.models.load_model(self.MODEL_PATH)
            print("[INFO] MODEL LOADED...")
        else:
            # input_shape = (look_back, 1)
            input_shape = (num_of_features, look_back)
            model = Sequential()
            model.add(
                LSTM(32, activation="relu", input_shape=input_shape))
            # model.add(
            #     Conv1D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu",
            #            input_shape=input_shape))
            # lstm_model.add(Dropout(0.1))
            model.add(Dropout(0.2))

            model.add(Dense(num_of_features, activation='relu'))

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='loss', patience=15, verbose=1)
        # callbacks=[early_stop]
        history = model.fit(trainX, trainY, epochs=EPOCHS, verbose=1, validation_data=(valX, valY))

        model.save(self.MODEL_PATH)
        print("[INFO] MODEL SAVED...")

        trainPredict = model.predict(trainX)

        valPredict = model.predict(valX)

        testPredict = model.predict(testX)


        # testR2 = r2_score(testY[:, 0], testPredict[:, 0])
        # print('Test R2: %.2f ' % (testR2))
        # valR2 = r2_score(valY[:, 0], valPredict[:, 0])
        # print('Val R2: %.2f ' % (valR2))
        # trainR2 = r2_score(trainY[:, 0], trainPredict[:, 0])
        # print('Train R2: %.2f ' % (trainR2))

        testR2 = r2_score(testY, testPredict)
        print('Test R2: %.2f ' % (testR2))
        valR2 = r2_score(valY, valPredict)
        print('Val R2: %.2f ' % (valR2))
        trainR2 = r2_score(trainY, trainPredict)
        print('Train R2: %.2f ' % (trainR2))

        feature_i = 0

        plt.plot(test.index[look_back+1:], testY[:, feature_i].ravel(), label="Test_obs")
        plt.plot(test.index[look_back+1:], testPredict[:, feature_i].ravel(), label="Test_pred")
        plt.plot(val.index[look_back+1:], valY[:, feature_i].ravel(), label="Val_obs")
        plt.plot(val.index[look_back+1:], valPredict[:, feature_i].ravel(), label="Val_pred")
        plt.plot(train.index[look_back+1:], trainY[:, feature_i].ravel(), label="Train_obs")
        plt.plot(train.index[look_back+1:], trainPredict[:, feature_i].ravel(), label="Train_pred")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def basicLSTM(self, stock_h):
        num_of_features = 4
        dataset = self.get_features(stock_h, num_of_features=num_of_features)
        train, test = self.split_dataset(dataset, "2016-01-01", initial_data_cut="2019-01-01")
        # train, test = self.split_dataset(dataset, "2018-01-01")
        train, val = self.split_dataset(train, "2012-01-01")

        look_back = 5
        trainX, trainY = self.create_dataset(train, look_back)
        valX, valY = self.create_dataset(val, look_back)
        testX, testY = self.create_dataset(test, look_back)

        trainX = np.reshape(trainX, (trainX.shape[0], num_of_features, trainX.shape[1]))
        valX = np.reshape(valX, (valX.shape[0], num_of_features, valX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], num_of_features, testX.shape[1]))

        early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

        SAVE = True

        if os.path.exists(self.MODEL_PATH) and SAVE:
            model = tensorflow.keras.models.load_model(self.MODEL_PATH)
        else:
            model = Sequential()
            model.add(LSTM(32, input_shape=(num_of_features, look_back)))
            model.add(Dropout(0.3))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        model.fit(trainX, trainY, epochs=25, batch_size=1, verbose=2, validation_data=(valX, valY),
                      callbacks=[early_stop])

        model.save(self.MODEL_PATH)

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
        # print('Train Score: %.2f RMSE' % (trainScore))
        # testScore = math.sqrt(mean_squared_error(testY, testPredict))
        # print('Test Score: %.2f RMSE' % (testScore))


        trainScore = r2_score(trainY, trainPredict)
        print('R2 Train Score: %.2f' % (trainScore))
        testScore = r2_score(testY, testPredict)
        print('R2 Test Score: %.2f' % (testScore))

        plt.plot(testY)
        plt.plot(testPredict)
        plt.show()

    def statefulLSTM(self, stock_h):
        dataset = self.get_features(stock_h, num_of_features=1)
        # train, test = split_dataset(dataset, "2019-01-01", initial_data_cut="2018-01-01")
        train, test = self.split_dataset(dataset, "2017-01-01")
        val, test = self.split_dataset(test, "2019-01-01")

        batch_size = 1
        look_back = 3
        EPOCHS = 25

        trainX, trainY = self.create_dataset(train, look_back)
        valX, valY = self.create_dataset(val, look_back)
        testX, testY = self.create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        # valX = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
        # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)


        # It can be used to reconstruct the model identically.
        if os.path.exists("models\stateful_lstm.h5"):
            model = tensorflow.keras.models.load_model("models\stateful_lstm.h5")
        else:
            model = Sequential()
            model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
            model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        for i in range(EPOCHS):
            print(f"[INFO] EPOCH: {i}/{EPOCHS}")
            model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False, validation_data=(valX, valY))
            # model.reset_states()

        model.save("models\stateful_lstm.h5")
        # model.save("stateful_lstm")

        # model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2, validation_data=(valX, valY),
        #           callbacks=[early_stop])
        trainPredict = model.predict(trainX, batch_size=batch_size)
        # model.reset_states()
        testPredict = model.predict(testX, batch_size=batch_size)

        # trainPredict = model.predict(trainX)
        # testPredict = model.predict(testX)

        # trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
        # print('Train Score: %.2f RMSE' % (trainScore))
        # testScore = math.sqrt(mean_squared_error(testY, testPredict))
        # print('Test Score: %.2f RMSE' % (testScore))
        #


        trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        plt.plot(testY)
        plt.plot(testPredict)
        plt.show()

        # # shift train predictions for plotting
        # trainPredictPlot = np.empty_like(dataset)
        # trainPredictPlot[:, :] = np.nan
        # trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
        # # shift test predictions for plotting
        # testPredictPlot = np.empty_like(dataset)
        # testPredictPlot[:, :] = np.nan
        # testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
        # # plot baseline and predictions
        # # plt.plot(scaler.inverse_transform(dataset))
        # plt.plot(trainPredictPlot)
        # plt.plot(testPredictPlot)
        # plt.show()

