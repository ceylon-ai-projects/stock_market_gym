from collections import deque

import numpy as np


class Agent():
    train_itr = 0

    def __init__(self,
                 name,
                 max_mem_len=1e3,
                 gamma=1e-2,
                 state_size=1,
                 action_size=3,
                 forget_rate=0.03,
                 epsilon=1e-4,
                 train_agent=True):
        self.name = name
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size
        self.train_agent = train_agent
        self.forget_rate = forget_rate
        self.__memory__ = deque(maxlen=int(max_mem_len))
        #
        self.after_init()

    def after_init(self):
        pass

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.action_size, size=1)
        else:
            q = self.predict_action(state)
            action = np.argmax(q[0])
        return action

    def train(self):
        pass

    def memorise(self, event):
        # print(event)
        # Memories Event
        self.__memory__.append(event)

        # When Memory full then do some actions
        if len(self.__memory__) == self.__memory__.maxlen:
            if self.train_agent:
                # Train
                self.train()
                self.train_itr += 1

            # Forget Memory events
            for f in range(int(len(self.__memory__) * self.forget_rate)):
                self.__memory__.popleft()

        # if random.random() > (1 - self.forget_rate):
        #     for f in range(int(len(self.__memory__) * self.forget_rate)):
        #         self.__memory__.popleft()

    def predict_action(self, state):
        pass
