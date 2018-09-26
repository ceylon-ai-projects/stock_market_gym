import os
import time

from keras.callbacks import TensorBoard
from keras.layers import Activation, Bidirectional, BatchNormalization
from rx import Observer
from ta import *

from agent.market_agent import Agent
from data.data_manager import get_data
from envrionment.market_environment import MarketEnv
from pattern_encoder import encode_column_to_range_index, pattern_to_action

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
# df['average_true_range']=average_true_range(data_csv.High,data_csv.Low,data_csv.)

df.drop(inplace=True)

model_path = dirname + "/model/"

"Equation => Change % (T+k) = { [Column Index (T+K) - Column Index (T)] /Column Index (T) } X 100 "


def change_values(dt_k, dt_t):
    return (dt_k - dt_t) * 100 / dt_t


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
        if index % 500 == 0:
            self.fig.savefig('results/foo_{}.png'.format(index))

        plt.pause(0.001)  # Note this correction


def r_score(y_true, y_pred):
    r = (y_true - y_pred)
    # print(r)
    r = ([1 if x == 0 else -1 for x in r])
    return r


class StockMarketCSV(MarketEnv):
    __current_index = 0

    def __after__init__(self):
        pass

    calculation_reward = 1
    calculation_reward_index = 1
    reward_cum = 0

    def __reward_func__(self, state, pre_state, action):
        # print("------")
        # print(state,pre_state,action)
        if pre_state is not None and len(pre_state) > 0:
            # print("--------\n")
            expected_result = pre_state[1]
            expected_result = [encode_column_to_range_index(er, i=i) for i, er in enumerate(expected_result)]

            exp_action = pattern_to_action(expected_result)

            # print(action)
            # print(expected_result)
            # print(exp_action)
            # print("--------\n")
            # action = action[0]
            # diff = exp_action - action

            reward = 1 if exp_action == action else -1

            if self.reward_player is not None and np.random.uniform(0, 1) > (1 - 0.3):
                self.reward_player.play(self.calculation_reward_index, (self.reward_cum / self.calculation_reward))

            if self.calculation_reward % 100 == 0:
                self.calculation_reward = 0
                self.reward_cum = 0

            self.calculation_reward += 1
            self.calculation_reward_index += 1
            self.reward_cum += reward

            return reward
        return -1


def action_shape(action):
    return action


class FxAgent(Agent):
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32,
                              write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None, embeddings_data=None)

    def __after_init__(self):
        # self.__model = load_model(model_path + 'fx_1-5_33_IT_6')
        self.build_model(input_shape=(1, self.state_size), output_dim=self.action_size)
        # pass

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.action_size, size=1)
        else:
            q = self.proactive(state)

            action = np.argmax(q[0])
            # print(q[0])
            # print(action)
        return action

    def proactive(self, state):
        state = state[0]
        state = np.reshape(state, (1, 1, state.shape[0]))
        action = self.__model.predict(state)
        return action_shape(action)

    def pre_process(self, data_list):
        data_list = np.array(data_list)
        data_list = data_list[:, 4:5]
        data_ar = np.reshape(data_list, (data_list.shape[0]))

        # print("------")
        #
        # print(data_ar)
        # print(data_ar[-self.future_state:(-self.future_state + 1)])
        # print(data_ar[-(self.future_state - 1):])
        # print(data_ar[:(-self.future_state)])

        data_state = change_values(data_ar[:(-self.future_state)], data_ar[-self.future_state:(-self.future_state + 1)])

        data_observation = change_values(data_ar[-(self.future_state - 1):],
                                         data_ar[-self.future_state:(-self.future_state + 1)])
        # print(data_state)
        # print(data_observation)
        # time.sleep(1)
        # print("------")
        return (data_state, data_observation)

    def build_model(self, input_shape, output_dim):
        from keras import Sequential
        from keras.layers import Dense, LSTM
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(12, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(12)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(output_dim))
        model.add(Activation("softmax"))
        model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
        self.__model = model

    def train(self, state, action):
        # state = state[0]
        # print(state)
        state = np.reshape(state, (state.shape[0], 1, state.shape[1]))
        # print(action.shape)
        action = np.reshape(action, (action.shape[0], self.action_size))

        history = self.__model.fit(state, action, epochs=1, verbose=0, callbacks=[self.tensorboard])
        self.print_random_data(history)

    def update_kb(self):
        super().update_kb()
        prefix = self.train_update_terms % 100
        self.__model.save(model_path + '' + self.name + "_{}".format(prefix))

    def __policy_update__(self, state, action, reward, next_state, done):
        target = np.zeros((self.action_size))
        # print(reward)
        # print(reward + self.gamma * np.argmax(self.act(next_state)))
        target[action] = reward + self.gamma * np.argmax(self.act(next_state))
        # print(target)
        return target


agent = FxAgent("fx_1-5",
                state_size=state_size,
                action_size=action_size,
                future_state=future_state,
                replay_size=3.2e2,
                train_alpha_value=int(10), random_print_rate=1e-5)

market = StockMarketCSV(pair_name=pair_name,
                        freq=interval,
                        agent=agent,
                        env_play_speed=action_size,
                        action_size=action_size,
                        reward_player=RewardPlayer(),
                        env_memory_length=state_size + future_state)


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


# iter=100
#
# for i in range(i)
csv_streamer = CSVDataStream()
# print(len(data_csv))
# print(data_csv.head(12))
# print(data_csv.tail(12))
for index, data_row in data_csv.iterrows():
    csv_streamer.on_next(data_row.values)
csv_streamer.on_completed()

done = time.time()
elapsed = done - start
market.summary()
print("Processing Time - {} sec".format(elapsed))
plt.show()
