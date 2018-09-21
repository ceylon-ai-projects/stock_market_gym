from rx import Observer

from agent.market_agent import Agent
from data.data_manager import get_data
from envrionment.market_environment import StockMarketCSV

pair_name = "EURUSD"
interval = 1

data_csv = get_data(pair_name, interval)
agent = Agent("fx_1-5")

market = StockMarketCSV(pair_name=pair_name,
                        freq=interval,
                        agent=agent,
                        env_play_speed=4,
                        env_memory_length=10)


class CSVDataStream(Observer):

    def on_next(self, value):
        market.feed_data(value)

    def on_error(self, error):
        print("<<<<<<-ERROR->>>>>")
        print(error)
        print("<<<<<<>>>>>>>>>>")

    def on_completed(self):
        market.finish()
        print("<<<<<<-COMPLETED->>>>>")


csv_streamer = CSVDataStream()
for index, data_row in data_csv.iterrows():
    csv_streamer.on_next(data_row.values)
csv_streamer.on_completed()
