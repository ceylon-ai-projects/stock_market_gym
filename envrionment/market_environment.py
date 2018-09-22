import copy
from collections import deque


# from sklearn.preprocessing import normalize





class MarketEnv():
    __current_state__ = None
    __market_current_step__ = 0
    __agents = []

    def __init__(self, pair_name, freq, env_memory_length=3, env_play_speed=1,
                 action_size=3,
                 window_length=1,
                 data_pre_processor=None,
                 reward_player=None,
                 agent=None):
        self.reward_player = reward_player
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
        self.window_length = window_length
        self.data_pre_processor = data_pre_processor
        self.__after__init__()

    def __after__init__(self):
        pass

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
        self.__last_action__ = self.agent.act(self.__current_state__)

        self.agent.memorize(__previous_state__, self.__last_action__,
                            self.__last_reward__, __current_state__, done)

    def finish(self):
        # self.agent.memorize(self.__current_state__, self.__last_action__,
        #                     self.__last_reward__, [], True)
        pass

    def summary(self):
        print(self.agent.summary())
        print("Data Steps {}".format(self.__market_current_step__))


'''

'''



