"""
Trains an epsilon-greedy QR-DQN agent on the gridworld

"""
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
from environments.gridworld import Gridworld

from agents.qrdqn.qrdqn import QRDQN
from agents.common.networks.mlp import MLP

world_height = 2
world_width = 5
wind_probability = 0.05
fall_reward = 0
success_reward = 10
nb_steps = 10000

for i in range(1):

    env = Gridworld(
        world_height=world_height,
        world_width=world_width,
        wind_probability=wind_probability,
        fall_reward=fall_reward,
        success_reward=success_reward
    )

    notes = "Height: {}, Width: {}, Wind Prob: {}, Fall reward: {}, Success reward: {}".format(
        str(world_height),
        str(world_width),
        str(wind_probability),
        str(fall_reward),
        str(success_reward)
        )

    agent = QRDQN( env,
                    MLP,
                    n_quantiles=50,
                    kappa=0,
                    weight_scale=3,
                    replay_start_size=500,
                    replay_buffer_size=10000,
                    gamma=1,
                    update_target_frequency=100,
                    minibatch_size=64,
                    learning_rate=2e-3,
                    initial_exploration_rate=1,
                    final_exploration_rate=0.05,
                    final_exploration_step=2000,
                    adam_epsilon=1e-8,
                    update_frequency=1,
                    logging=True,
                    log_folder_details="Gridworld-QRDQN",
                    notes=notes)

    agent.learn(timesteps=nb_steps, verbose=True)