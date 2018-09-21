from collections import deque

import numpy as np


class MarketEnv():
    __current_state__ = None
    __market_current_step__ = 0

    def __init__(self, pair_name, freq, env_memory_length=3, env_play_speed=1):
        self.env_play_speed = env_play_speed
        self.env_memory_length = env_memory_length
        self.freq = freq
        self.pair_name = pair_name
        self.__current_state__ = []
        self.__data__set__ = deque()

    def feed_data(self, data):
        if len(self.__data__set__) >= self.env_memory_length:
            self.__current_state__ = np.array(list(self.__data__set__))
            for i in range(self.env_play_speed):
                if len(self.__data__set__) >= 0:
                    self.__data__set__.popleft()
        self.__data__set__.append(data)
        self.__market_current_step__ += 1

    def __data_stream__(self):
        pass

    def __reward_func__(self, state, pre_state, action):
        pass

    def __reveal_next_state__(self):
        return self.__current_state__

    def do_action(self, action):
        done = False
        info = None
        previouse_state = self.__current_state__
        self.__reveal_next_state__()  # Change Current State
        reward = self.reward_func(self.__current_state__, previouse_state, action)  # Do Action And get Rewards

        if self.__current_state__ == None:
            done = True
        return self.__current_state__, reward, done, info

    def reset(self):
        self.__current_state__ = []


'''

'''

from data.data_manager import get_data


class StockMarketCSV(MarketEnv):
    __current_index = 0

    def __init__(self, pair_name, freq, window_length=1, data_pre_processor=None, env_memory_length=3,
                 env_play_speed=1):
        super().__init__(pair_name, freq, env_memory_length=env_memory_length, env_play_speed=env_play_speed)
        self.window_length = window_length
        self.data_pre_processor = data_pre_processor

    '''
    Header => ( Date_Time, Open, High, Low, Vol)
    '''

    def __data_stream__(self):
        self.__data__ = get_data(pair=self.pair_name, interval=self.freq)
        self.__data__ = self.data_pre_processor(self.__data__)

    def __reveal_next_state__(self):
        state = self.__data__[self.__current_index:(self.__current_index + self.window_length)]
        self.__current_state__ = state
        return super().__reveal_next_state__()

    def __reward_func__(self, state, pre_state, action):
        pass
