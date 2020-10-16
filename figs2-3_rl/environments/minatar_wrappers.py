"""
Wrapper to make the MinAtar environment compatible with this repo
"""

import minatar
import torch


class MinatarWrapper(minatar.Environment):

    def __init__(self, env):

        self.env = env

        class action_space():
            """
            For discrete actions, action_space.n is the number of actions.
            2 actions by default, as in Cartpole.
            """
            def __init__(self, num_actions): 
                self.n = num_actions

        class observation_space():
            """
            For discrete actions, action_space.n is the number of actions.
            2 actions by default, as in Cartpole.
            """
            def __init__(self, env): 
                self.shape = env.state_shape()

        self.action_space = action_space(self.env.num_actions())
        self.observation_space = observation_space(self.env)

    def reset(self):
        self.env.reset()
        return self.env.state()

    def step(self, action):
        reward, done = self.env.act(action)
        next_state = self.env.state()
        info = None
        return next_state, reward, done, info

    def seed(self, seed):
        """
        Set random seed of the environment
        """
