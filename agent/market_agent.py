from collections import deque
import numpy as np
import random


class Agent():
    train_itter_number = 0
    train_update_terms = 0
    __print_itr = 0  # Print Iteration Count

    def __init__(self, name, state_size=None, is_eval=False,
                 replay_size=100,
                 train_alpha_value=4,
                 max_memory_length=int(1e3),
                 gamma=0.95, epsilon=1, epsilon_min=1e-4,
                 epsilon_decay=0.9,
                 action_size=3,
                 batch_size=72, random_print_rate=10):
        self.random_print_rate = random_print_rate
        self.train_alpha_value = train_alpha_value
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

        self.__after_init__()

    def __after_init__(self):
        pass

    def reset(self, hard_rest=False, state_size=None, is_eval=False,
              replay_size=100,
              max_memory_length=int(1e3),
              train_alpha_value=4,
              gamma=0.95, epsilon=1, epsilon_min=1e-4,
              epsilon_decay=0.9,
              action_size=3,
              batch_size=72):
        self.train_alpha_value = train_alpha_value
        self.replay_size = replay_size
        self.is_eval = is_eval
        self.state_size = state_size

        self.action_size = action_size  # Sell, Buy,Wait
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory_length)

        if hard_rest:
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay

    def pre_process(self, data):
        return data

    def proactive(self, state, reward):
        return np.random.rand(self.action_size)

    def train(self, train_x, train_y):
        # print("Train")
        self.train_itter_number += 1

    def update_kb(self):
        # print("Update KB")
        self.train_update_terms += 1

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) >= self.replay_size or done:
            self.__exp_play()

            # Forgeting Events#
            memory_forget = random.randint(0, self.train_alpha_value)
            memory_forget = 1 if done else memory_forget
            for i in range(memory_forget):
                if len(self.memory) > 0:
                    self.memory.popleft()

    def __exp_play(self):
        mini_batch = []
        l = len(self.memory)
        if l >= self.replay_size:
            for i in range(l - self.batch_size + 1, l):
                mini_batch.append(self.memory[i])
        else:
            mini_batch = self.memory

        train_x = []
        train_y = []
        for state, action, reward, next_state, done in mini_batch:
            train_x.append(state)
            train_y.append(action)

            if random.uniform(0, 1) <= self.random_print_rate:
                self.__print__(state, action, reward, next_state, done)

            if done:
                print("Last Steps")
                print(self.train_itter_number, self.train_update_terms)

        self.train(np.array(train_x), np.array(train_y))
        self.train_itter_number += 1

        self.update_kb()
        self.train_update_terms += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_kb()

    def print_random_data(self, data):
        if random.uniform(0, 1) <= self.random_print_rate:
            print("<<<<<<<<<<<<<MESSAGE>>>>>>>>>>>>")
            print(data)
            print("<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>")

    def __print__(self, state, action, reward, next_state, done):
        print("<<<<<<<<<<<<<START- {}>>>>>>>>>>>>".format(self.__print_itr))
        print("State ", state)
        print("Action ", action)
        print("Reward ", reward)
        print("Next State ", next_state)
        print("Done ", done)
        print("<<<<<<<<<<<<END>>>>>>>>>>>>>")
        self.__print_itr += 1

    def summary(self):
        print(">>>>>SUMMARY<<<<<<")
        print("Train itr- {}".format(self.train_itter_number))
        print("Train update term- {}".format(self.train_update_terms))
