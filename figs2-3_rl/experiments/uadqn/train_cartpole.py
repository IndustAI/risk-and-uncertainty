#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains a UQ-DQN agent on Cartpole

"""
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents.uadqn.uadqn import UADQN
from agents.common.networks.mlp import MLP

notes = "This is a test run"

env = gym.make("CartPole-v0")

nb_steps = 5000

agent = UADQN(
    env,
    MLP,
    n_quantiles=20,
    weight_scale=3,
    noise_scale=1,
    epistemic_factor=1,
    aleatoric_factor=0,
    kappa=10,
    replay_start_size=50,
    replay_buffer_size=50000,
    gamma=0.99,
    update_target_frequency=50,
    minibatch_size=32,
    learning_rate=1e-3,
    adam_epsilon=1e-8,
    update_frequency=1,
    logging=True,
    log_folder_details="Cartpole-UADQN",
    notes=notes)

agent.learn(timesteps=nb_steps, verbose=True)
