from collections import deque
import numpy as np
import random


class Agent():

    def __init__(self, name, state_size=None, is_eval=False,
                 replay_size=100,
                 max_memory_length=int(1e3),
                 gamma=0.95, epsilon=1, epsilon_min=1e-4,
                 epsilon_decay=0.9,
                 action_size=3,
                 batch_size=72):
        self.replay_size = replay_size
        self.name = name
        self.is_eval = is_eval
        self.state_size = state_size

        self.action_size = action_size  # Sell, Buy,Wait
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory_length)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            values = random.randrange(self.action_size)
        else:
            state = np.reshape(state, (1, state.shape[1],))
            options = self.predict(state)
            values = np.argmax(options)
        return values

    def proactive(self, state, reward):
        return np.random.rand(self.action_size)

    def train(state, traget_f):
        print("Train")

    def update_kb(self):
        print("Update KB")

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) >= self.replay_size or done:
            self.__exp_play()
            memory_forget = random.randint(0, self.replay_size - 1)
            memory_forget = 1 if done else memory_forget
            for i in range(memory_forget):
                self.memory.popleft()

    def __exp_play(self):
        # print("play_on_batch....")
        mini_batch = []
        l = len(self.memory)

        for i in range(l - self.batch_size + 1, l):
            mini_batch.append(self.memory[i])

        # self.memory.clear()

        for state, action, reward, next_state, done in mini_batch:
            print("--Mini Batch--")
            print(state, action, reward, next_state, done)
            if done:
                print("Last Steps")

        self.update_kb()
