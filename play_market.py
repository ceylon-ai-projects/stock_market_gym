import copy
import os
import random
import time

from keras.engine.saving import load_model
from ta import *

from data.data_manager import get_data
from envrionment.market_env import Agent, MarketEnvironment, DataAgent

import matplotlib.pyplot as plt

from pattern_encoder import encode_column_to_range_index, decode_column_to_int

dirname = os.path.dirname(__file__)
start = time.time()

dirname = os.path.dirname(__file__)
model_path = dirname + "/model/"

pair_name = "EURUSD"
interval = 1

future_state = 4
state_size = 3
action_size = 3
considering_steps = 7

rsi_range = range(14, 15)
tsi_range = range(14, 15)
emi_range = range(12, 13)
aroon_range = range(25, 26)
dpo_range = range(20, 26)

data_csv = get_data(pair_name, interval)
print(data_csv.head())
df = pd.DataFrame()
close_s = data_csv.Close
df['C'] = close_s

for rsi_i in rsi_range:
    df['RSI({})'.format(rsi_i)] = rsi(data_csv.Close)

for atr_i in tsi_range:
    df['ATR({})'.format(atr_i)] = average_true_range(data_csv.High, data_csv.Low, data_csv.Close, n=atr_i)

for ema_i in emi_range:
    df['exp({})'.format(ema_i)] = ema(data_csv.Close, ema_i)

for aron_i in aroon_range:
    df['arn_d({})'.format(aron_i)] = aroon_down(data_csv.Close, n=aron_i)
    df['arn_u({})'.format(aron_i)] = aroon_up(data_csv.Close, n=aron_i)

for dpo_i in dpo_range:
    df['dpo({})'.format(dpo_i)] = ema(data_csv.Close, dpo_i)

# Pattern
df['P'] = pd.Series(np.full(df['C'].values.shape, ''))
for back_step in range(-considering_steps, 0):
    series = (close_s.shift(-back_step) - close_s) * 100 / close_s
    series = series.apply(encode_column_to_range_index)
    df['P'] = df['P'] + series
df.dropna(inplace=True)
df['P'] = df.P.apply(decode_column_to_int)

state_size = df.shape[1]

print(df.describe())


class CsvDataAgent(DataAgent):

    def get_data(self, step):
        state = df.iloc[step, :].values
        done = False
        if len(df) == step + 1:
            done = True
        # print(state,step)
        # time.sleep(1)
        return state, done


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


class MarketAgent(Agent):

    def build_model(self, input_shape):
        from keras import Sequential
        from keras.layers import LSTM, Bidirectional, Dense, Activation
        model = Sequential()
        model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        # model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(self.action_size))
        model.add(Activation("softmax"))
        model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
        self.model = model
        print(self.model.summary())

    def after_init(self):
        # self.build_model((1, self.state_size))
        self.model = load_model(model_path + "fx_agent_rl__0")

    def update_policy(self, action, reward, state, state_next):
        target = np.zeros((self.action_size))
        target[action] = reward + self.gamma * np.argmax(self.act(state_next))
        return target

    def train(self):
        memory = np.array(copy.deepcopy(self.__memory__))
        np.random.shuffle(memory)

        train_x = []
        train_y = []
        for state, next_state, action, reward in memory:
            train_x.append(state)
            # print(state)
            target = self.update_policy(action, reward, state, next_state)
            # print(state, target)
            train_y.append(target)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_x = np.reshape(train_x, (train_x.shape[0], 1, self.state_size))
        # print(train_y)
        # print(train_y.shape)
        self.model.fit(train_x, train_y, verbose=0, batch_size=32)

        print(self.train_itr)

        if self.train_itr % 5 == 0:
            prefix = self.train_itr % 5
            self.model.save(model_path + '' + self.name + "_{}".format(prefix))

    def predict_action(self, state):
        # print(state)
        state = np.reshape(state, (1, 1, state.shape[0]))
        return self.model.predict(state)
        # return np.random.rand(1, self.action_size)


def dif_to_action(diff):
    if diff < 0:
        return 0  # Sell
    elif diff == 0:
        return 1  # Stay
    else:
        return 2  # Buy


class EURUSDMarket(MarketEnvironment):
    reward_player = None  # RewardPlayer()

    reward_index = 0
    reward_cum_index = 1
    total_rewards = 0

    def reward_func(self, state_t, state_t_next, action_t):
        # print(state_t, state_t_next, action_t)
        diff = state_t[0] - state_t_next[0]
        actual_action = dif_to_action(diff)
        reward = 1 if actual_action == action_t else -1

        # print(reward, action_t, state_t[0], state_t_next[0], diff, actual_action)

        self.total_rewards += reward
        reward_value = (self.total_rewards / self.reward_cum_index)
        if self.reward_player is not None and np.random.rand() > (1 - 0.05):
            self.reward_player.play(self.reward_index, reward_value)

        if self.reward_index % 100 == 0:
            self.reward_cum_index = 1
            self.total_rewards = 0

        self.reward_index += 1
        self.reward_cum_index += 1
        return reward


data_agent = CsvDataAgent()
prediction_agent = MarketAgent("fx_agent_rl_", state_size=state_size, max_mem_len=2.5e3, forget_rate=0.6)

market_env = EURUSDMarket(agent=prediction_agent, data_agent=data_agent)
while market_env.stop_command is not True:
    market_env.play()
