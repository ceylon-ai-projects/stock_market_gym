





class DataAgent():

    def get_data(self, step):
        return []


class MarketEnvironment():
    __t = 0
    t_last = 0
    last_state = None
    last_action = None
    stop_command = False
    agent_act = False

    def __init__(self, agent, data_agent, action_evaluate_steps=3):
        self.action_evaluate_steps = action_evaluate_steps
        self.agent = agent
        self.data_agent = data_agent

    def __state(self, time=0):
        return self.data_agent.get_data(time)

    def reward_func(self, state_t, state_t_next, action_t):
        reward = 0
        return reward

    def play(self):
        self.__loop()

    def __loop(self):
        # Get State time t
        state_t, done = self.__state(self.__t)
        # Get Action from Agent
        if self.agent_act is False:
            action = self.agent.act(state_t)
            self.last_action = action
            self.last_state = state_t
            self.t_last = self.__t
            self.agent_act = True

        # print(self.__t, self.action_evaluate_steps, self.t_last)
        # Calculate Reward and t_next
        if self.t_last + self.action_evaluate_steps == self.__t:
            reward = self.reward_func(self.last_state, state_t, self.last_action)
            # Memorize Agent his exp
            if self.agent.train:
                # Event = state_t,state_t_next,action_t,reward_t,
                self.agent.memorise(
                    (self.last_state, state_t, self.last_action, reward)
                )
            # Update Env Variables

            self.agent_act = False
        self.__t += 1
        if done:
            self.__t = 0
