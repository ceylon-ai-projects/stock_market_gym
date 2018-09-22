import os
import time

from keras.layers import Dropout
from rx import Observer

from agent.market_agent import Agent
from data.data_manager import get_data
from envrionment.market_environment import StockMarketCSV

dirname = os.path.dirname(__file__)
start = time.time()

pair_name = "EURUSD"
interval = 1

data_csv = get_data(pair_name, interval)

model_path = dirname + "/model/"

"Equation => Change % (T+k) = { [Column Index (T+K) - Column Index (T)] /Column Index (T) } X 100 "


def change_values(dt_t_k, dt_t):
    return ((dt_t_k - dt_t) * 100) / dt_t


class ModelEvaluator(Observer):

    def on_next(self, value):
        pass


import matplotlib.pyplot as plt
import numpy as np


class RewardPlayer():
    plt.ion()  ## Note this correction
    fig = plt.figure()

    # plt.axis([0, 1000, 0, 1])

    def play(self, index, value):
        plt.scatter(index, value)
        plt.show()
        plt.pause(0.001)  # Note this correction


class FxAgent(Agent):

    def __after_init__(self):
        # self.__model=load_model(model_path + '' + self.name)
        self.build_model(input_shape=(1, self.state_size), output_dim=self.action_size)
        pass

    def pre_process(self, data_list):
        data_list = np.array(data_list)
        # print(data_list[:, [0, 4]])
        data_list = data_list[:, 4:5]

        # data_tk = data_list[-1:]
        #
        data_ar = data_list
        # for data in data_list:
        #     data_ar.append(change_values(data[0], data_tk[0]))

        data_ar = np.array(data_ar)
        # print(data_ar)
        # data_ar = normalize(data_ar)
        # print(data_ar)
        data_ar = np.reshape(data_ar, (data_ar.shape[0]))

        return data_ar

    def build_model(self, input_shape, output_dim):
        from keras import Sequential
        from keras.layers import Dense, LSTM
        model = Sequential()
        model.add(LSTM(256, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(256))
        model.add(Dense(128))
        model.add(Dense(64))
        model.add(Dense(32))
        model.add(Dense(output_dim))
        model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
        self.__model = model

    def proactive(self, state):
        # print(state.shape)
        state = np.reshape(state, (1, 1, state.shape[0]))
        action = self.__model.predict(state)
        # action_divident = np.abs(action)
        # print("Action")
        # print(action)
        # action = np.divide(action, action_divident)
        # print(action)
        return action

    def train(self, state, action):
        state = np.reshape(state, (state.shape[0], 1, state.shape[1]))
        # print(action.shape)
        action = np.reshape(action, (action.shape[0], self.action_size))
        history = self.__model.fit(state, action, epochs=1, verbose=0, validation_split=0.05)
        self.print_random_data(history)

    def update_kb(self):
        super().update_kb()
        prefix = self.train_update_terms % 100
        self.__model.save(model_path + '' + self.name + "_{}".format(prefix))


agent = FxAgent("fx_1-5",
                state_size=36,
                action_size=5,
                replay_size=1e2,
                train_alpha_value=int(10), random_print_rate=1e-5)

market = StockMarketCSV(pair_name=pair_name,
                        freq=interval,
                        agent=agent,
                        env_play_speed=5,
                        action_size=5,
                        reward_player=RewardPlayer(),
                        env_memory_length=36)


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
print(len(data_csv))
for index, data_row in data_csv.iterrows():
    csv_streamer.on_next(data_row.values)
csv_streamer.on_completed()

done = time.time()
elapsed = done - start
market.summary()
print("Processing Time - {} sec".format(elapsed))
