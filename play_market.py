import copy
import os
import time

import matplotlib.pyplot as plt

from ta import *

from agent.env_agents import Agent
from data.data_manager import get_data
from envrionment.market_env import MarketEnvironment, DataAgent
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
considering_steps = 15

rsi_range = [14, 29, 58, 100]
tsi_range = [14, 29, 58, 100]
emi_range = [9, 11, 20, 100]
aroon_range = [9, 13, 29, 50]
dpo_range = [4, 5, 13, 35]

data_csv = get_data(pair_name, interval)
data_csv['Ct'] = data_csv.Close.shift(considering_steps)
data_csv.dropna(inplace=True)
print(data_csv.head())
df = pd.DataFrame()
close_s = data_csv.Close
df['C'] = close_s


def process_change_series(close_s, step_back_s):
    series = (step_back_s - close_s) * 100 / close_s
    # print(series[-1:])
    return series.apply(encode_column_to_range_index)


for rsi_i in rsi_range:
    df['RSI({})'.format(rsi_i)] = rsi(close_s)

for atr_i in tsi_range:
    df['ATR({})'.format(atr_i)] = average_true_range(data_csv.High, data_csv.Low, close_s, n=atr_i)

for ema_i in emi_range:
    df['exp({})'.format(ema_i)] = ema(close_s, ema_i)

for aron_i in aroon_range:
    df['arn_d({})'.format(aron_i)] = aroon_down(close_s, n=aron_i)
    df['arn_u({})'.format(aron_i)] = aroon_up(close_s, n=aron_i)

for dpo_i in dpo_range:
    df['dpo({})'.format(dpo_i)] = ema(data_csv.Close, dpo_i)

# Pattern
series = (close_s.shift(1) - close_s) * 100 / close_s
series = series.apply(encode_column_to_range_index)
df['P1'] = series
df['P2'] = series
df['P3'] = series
df['P4'] = series
#
for back_step in range(2, (considering_steps - 1) + 1):
    df['P1'] += process_change_series(close_s, close_s.shift(back_step))

#
for back_step in range(2, 5):
    df['P2'] += process_change_series(close_s, close_s.shift(back_step))

#
for back_step in range(2, 4):
    df['P3'] += process_change_series(close_s, close_s.shift(back_step))

for back_step in range(2, 10):
    df['P4'] += process_change_series(close_s, close_s.shift(back_step))
# print(df['P'])
df.dropna(inplace=True)
# print(df.values[-10:, -4:])
df['P1'] = df.P1.apply(decode_column_to_int)
df['P2'] = df.P2.apply(decode_column_to_int)
df['P3'] = df.P3.apply(decode_column_to_int)
df['P4'] = df.P3.apply(decode_column_to_int)
# print(df.values[-10:, -4:])
state_size = df.shape[1]

print(df.tail())
print(data_csv.tail())


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
        plt.scatter(index, value, s=2)
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
        self.build_model((1, self.state_size))
        # from keras.engine.saving import load_model
        # self.model = load_model(model_path + "fx_agent_rl__4")

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
            target = self.update_policy(action, reward, state, next_state)
            train_y.append(target)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_x = np.reshape(train_x, (train_x.shape[0], 1, self.state_size))
        self.model.fit(train_x, train_y, verbose=0, batch_size=512, epochs=50)

        if self.itr_index % 5 == 0:
            prefix = self.itr_index % 5
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
    reward_player = RewardPlayer()

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
        if self.reward_player is not None and np.random.rand() > (1 - 0.5):
            self.reward_player.play(self.reward_index, reward_value)

        if self.reward_index % 100 == 0:
            self.reward_cum_index = 1
            self.total_rewards = 0

        self.reward_index += 1
        self.reward_cum_index += 1
        return reward


data_agent = CsvDataAgent()
prediction_agent = MarketAgent("fx_agent_rl_",
                               state_size=state_size,
                               gamma=0.5,
                               epsilon_decay=0.9,
                               max_mem_len=5e2,
                               epsilon_min=1e-7,
                               forget_rate=0.7)

market_env = EURUSDMarket(agent=prediction_agent, data_agent=data_agent)
while market_env.stop_command is not True:
    market_env.play()
