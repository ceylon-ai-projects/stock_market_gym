import copy
from collections import deque

import numpy as np


class MarketEnv():
    __current_state__ = None
    __market_current_step__ = 0
    __agents = []

    def __init__(self, pair_name, freq, env_memory_length=3, env_play_speed=1,
                 action_size=3,
                 agent=None):
        self.__action_size__ = action_size
        self.agent = agent
        self.env_play_speed = env_play_speed
        self.env_memory_length = env_memory_length
        self.freq = freq
        self.pair_name = pair_name
        self.__current_state__ = []
        self.__previous_state__ = []
        self.__last_reward__ = []
        self.__last_action__ = []
        self.__data__set__ = deque()

    def feed_data(self, data, last=False):

        # Market in action with enough data
        if len(self.__data__set__) >= self.env_memory_length:
            __previous_state__ = copy.deepcopy(self.__current_state__)
            # print(__previous_state__)
            self.__current_state__ = self.agent.pre_process(list(self.__data__set__))  # Preprocess Data Window
            self.__process_state(__previous_state__, self.__current_state__, done=last)

            for i in range(self.env_play_speed):
                if len(self.__data__set__) >= 0:
                    self.__data__set__.popleft()

        self.__data__set__.append(data)
        self.__market_current_step__ += 1
        # print(data)

    def __reward_func__(self, state, pre_state, action):
        pass

    def register_agent(self, agent):
        self.__agents.insert(agent)

    def reset(self):
        self.__current_state__ = []
        self.__previous_state__ = []
        self.__last_reward__ = []
        self.__last_action__ = []
        self.__data__set__ = deque()

    def __process_state(self, __previous_state__, __current_state__, done):
        self.__reward_func__(__current_state__,
                             pre_state=__previous_state__, action=self.__last_action__)
        self.__last_action__ = self.agent.proactive(self.__current_state__, self.__last_reward__)

        self.agent.memorize(__previous_state__, self.__last_action__,
                            self.__last_reward__, __current_state__, done)

    def finish(self):
        self.agent.memorize(self.__current_state__, self.__last_action__,
                            self.__last_reward__, [], True)

    def summary(self):
        print(self.agent.summary())
        print("Data Steps {}".format(self.__market_current_step__))


'''

'''
class StockMarketCSV(MarketEnv):
    __current_index = 0

    def __init__(self, pair_name, freq, window_length=1,
                 action_size=3, agent=None, data_pre_processor=None, env_memory_length=3,
                 env_play_speed=1):
        super().__init__(pair_name, freq,
                         agent=agent,
                         action_size=action_size,
                         env_memory_length=env_memory_length, env_play_speed=env_play_speed)
        self.window_length = window_length
        self.data_pre_processor = data_pre_processor

    def __reward_func__(self, state, pre_state, action):
        if pre_state is not None and len(pre_state) > 0:
            # print("----PRE-STATE----")
            # print(pre_state[:, :1])
            # print("----STATE----")
            # print(state[:, :1])
            # print("----ACTION----")
            # print(action)
            self.__last_reward__ = np.random.rand(self.__action_size__)
