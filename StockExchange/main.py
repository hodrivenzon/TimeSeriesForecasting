import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import yfinance as yf

from StockExchange.LSTMModels import simpleLSTM

def plot_param(stock_h):
    plt.plot(stock_h["Close"])
    # plt.plot(stock_h["Open"])
    plt.show()


if __name__ == '__main__':

    # df = pd.read_csv(r"datasets\stockExchangeData\prices-split-adjusted.csv")

    msft = yf.Ticker("MSFT")
    tsla = yf.Ticker("TSLA")

    msft_h = msft.history(period="max")
    tsla_h = tsla.history(period="max")

    # plot_param(tsla_h)

    # basic_lstm(tsla_h)
    # simpleLSTM().basicLSTM(tsla_h)
    simpleLSTM().LSTM_CNN(tsla_h)
    # simpleLSTM().statefulLSTM(tsla_h)


    # plot_param(msft_h)
    # print("MSFT:")
    # print(msft_h)
    # print("TSLA:")
    # print(tsla_h)

    print()