import os
import time

from ta import *

from data.data_manager import get_data

dirname = os.path.dirname(__file__)
start = time.time()

pair_name = "EURUSD"
interval = 1

future_state = 4
state_size = 9
action_size = 3

data_csv = get_data(pair_name, interval)
df = pd.DataFrame()
df['Close'] = data_csv.Close
df['RSI'] = rsi(data_csv.Close)
df['average_true_range'] = average_true_range(data_csv.High, data_csv.Low, data_csv.Close)





