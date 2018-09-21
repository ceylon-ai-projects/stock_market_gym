import os

import pandas as pd
import requests

from config.config_manager import get_firebase_storage

dirname = os.path.dirname(__file__)


def get_real_time_data(from_symbol, to_symbol, interval):
    __key_ = "AW3Y3X9D1N0C1RYU"
    __link = "https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={from_symbol}&to_symbol={to_symbol}&interval={interval}min&apikey={key}&datatype=csv".format(
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        interval=interval,
        key=__key_
    )
    print(__link)
    data = pd.read_csv(__link, parse_dates=['timestamp'])
    data.columns = ['Date_Time', "Open", "High", "Low", "Close"]
    data['Date_Time'] = data['Date_Time'].dt.tz_localize('utc').dt.tz_convert('Asia/Colombo')
    data = data[::-1]  # Reverse Data
    print(data.tail())
    return data


def __get_data_csv(filepath, portion, data_length):
    names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Vol']
    data_csv = pd.read_csv(filepath, names=names,
                           parse_dates=[['Date', 'Time']])

    if data_length is not None:
        tot_lot = data_length
    else:
        tot_size = len(data_csv)
        # print(tot_size)
        tot_lot = int(tot_size * portion / 100)
    return data_csv.tail(tot_lot)


def __get_local_data(pair, interval, portion=90, data_length=None):
    file_path = "./store/{pair}/{pair}{interval}.csv".format(
        pair=pair, interval=interval)
    file_path = os.path.join(dirname, file_path)

    # print(tot_lot)
    return __get_data_csv(file_path, portion, data_length)


def get_data_chunk(pair, interval, chunk_size):
    file_path = "./store/{pair}/{pair}{interval}.csv".format(
        pair=pair, interval=interval)
    file_path = os.path.join(dirname, file_path)

    names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Vol']
    return pd.read_csv(file_path, names=names,
                       chunksize=chunk_size,
                       parse_dates=[['Date', 'Time']])


def __get_data_from_cloud(pair, interval, portion=90, data_length=None):
    bucket = get_firebase_storage()
    blob = bucket.blob('data/{pair_name}/{pair_name}{interval}.zip'.format(pair_name=pair, interval=interval))
    print(blob)
    direcotry_path = os.path.join(dirname, "./cloud/{}".format(pair))
    if os.path.exists(direcotry_path) is False:
        os.makedirs(direcotry_path)

    file_path = "{}/{}{}.zip".format(direcotry_path, pair, interval)

    with open(file_path, 'wb') as file_obj:
        blob.download_to_file(file_obj)

    return __get_data_csv(file_path, portion, data_length)


def get_data(pair, interval, portion=90, data_length=None, from_cloud=False):
    if from_cloud:
        return __get_data_from_cloud(pair, interval, portion, data_length)
    else:
        return __get_local_data(pair, interval, portion=portion, data_length=data_length)
