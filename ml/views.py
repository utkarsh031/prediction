from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import pandas_datareader as pdr

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from datetime import datetime
import datetime as dt

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def index(request):
    return render(request, 'index.html')

def get_historical(company):
    key = settings.ALPHA_VANTAGE_API_KEY
    end = dt.datetime.now()
    start = dt.datetime(end.year - 2, end.month, end.day)
    df = pdr.get_data_tiingo(company, api_key=key)
    df.to_csv(f'{company}.csv')
    df = pd.read_csv(f'{company}.csv')
    return df

def lstm_model(df):
    dataset_train = df.iloc[0:int(0.8 * len(df)), :]
    dataset_test = df.iloc[int(0.8 * len(df)):, :]

    training_set = df.iloc[:, 2:3].values

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - 7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_forecast = np.array(X_train[-1, 1:])
    X_forecast = np.append(X_forecast, y_train[-1])

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))

    import tensorflow as tf
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.layers import Dense, Dropout, LSTM # type: ignore

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=25, batch_size=32)

    real_stock_price = dataset_test.iloc[:, 2:3].values

    dataset_total = pd.concat((dataset_train['close'], dataset_test['close']), axis=0)
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
    testing_set = testing_set.reshape(-1, 1)
    testing_set = sc.transform(testing_set)

    X_test = []
    for i in range(7, len(testing_set)):
        X_test.append(testing_set[i - 7:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(real_stock_price, label='Actual Price')
    plt.plot(predicted_stock_price, label='Predicted Price')
    plt.legend(loc=4)
    plt.savefig(os.path.join(settings.STATIC_ROOT, 'LSTM.png'))
    plt.close(fig)

    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    forecasted_stock_price = model.predict(X_forecast)
    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
    lstm_pred = forecasted_stock_price[0, 0]

    return lstm_pred, error_lstm

@csrf_exempt
@require_POST
def insertintotable(request):
    if request.method == 'POST':
        nm = request.POST.get('nm', '')

        try:
            df = get_historical(nm)
        except Exception as e:
            return render(request, 'index.html', {'not_found': True, 'error_message': str(e)})

        today_stock = df.iloc[-1:]
        df = df.dropna()

        code_list = [nm] * len(df)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2

        lstm_pred, error_lstm = lstm_model(df)

        context = {
            'company': nm,
            'lstm_pred': round(lstm_pred, 2),
            'open_s': today_stock['open'].to_string(index=False),
            'close_s': today_stock['close'].to_string(index=False),
            'adj_close': today_stock['adjClose'].to_string(index=False),
            'high_s': today_stock['high'].to_string(index=False),
            'low_s': today_stock['low'].to_string(index=False),
            'vol': today_stock['volume'].to_string(index=False),
            'error_lstm': round(error_lstm, 2)
        }

        return render(request, 'results.html', context)
 