#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains a UA-DQN agent on the gridworld

"""
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
from environments.gridworld import Gridworld

from agents.uadqn.uadqn import UADQN
from agents.common.networks.mlp import MLP

ALEATORIC_FACTOR = 0.5

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

    agent = UADQN(
        env,
        MLP,
        n_quantiles=50,
        weight_scale=3,
        noise_scale=1,
        epistemic_factor=2,
        aleatoric_factor=ALEATORIC_FACTOR,
        kappa=0,
        replay_start_size=500,
        replay_buffer_size=10000,
        gamma=1,
        update_target_frequency=100,
        minibatch_size=64,
        learning_rate=2e-3,
        adam_epsilon=1e-8,
        biased_aleatoric=False,
        update_frequency=1,
        logging=True,
        log_folder_details="Gridworld-UADQN",
        notes=notes)

    agent.learn(timesteps=nb_steps, verbose=True)
