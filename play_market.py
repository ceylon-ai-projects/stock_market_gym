from rx import Observer

from data.data_manager import get_data
from env.market_environment import StockMarketCSV

pair_name = "EURUSD"
interval = 1

data_csv = get_data(pair_name, interval)

market = StockMarketCSV(pair_name=pair_name,
                        freq=interval,
                        env_play_speed=5,
                        env_memory_length=1000)


class CSVDataStream(Observer):

    def on_next(self, value):
        market.feed_data(value)

    def on_error(self, error):
        pass

    def on_completed(self):
        pass


csv_streamer = CSVDataStream()

for index, data_row in data_csv.iterrows():
    csv_streamer.on_next(data_row.values)
