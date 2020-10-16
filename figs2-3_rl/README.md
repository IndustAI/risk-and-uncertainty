# MinAtar experiments

Code used to train and test agents on the gridworld and the MinAtar testbed.

# Structure

This folder contains the following folders:

**agents**: the learning agents: UADQN (us), QRDQN, DQN, and Bootstrapped DQN
**environments**: the gridworld environment, and a wrapper for MinAtar
**experiments**: containing scripts to train and test agents.
**results**: by default, the output of scripts from the experiments folder is saved here
**plotting**: loads data from the results folder and plots it, yielding the figures in our paper

#Running experiments on the gridworld: example

To train a risk-averse UA-DQN agent on the gridworld, run the following command from this folder:

'''
python -m experiments.uadqn.train_gridworld
'''

A log folder will automatically be created in the results folder, which at the end of training will contain the trained agent's network weights, log data, and plots of the agent's performance. 

Once several seeds have been saved, the training curves and the cumulative number of falls can be plotted by running:

'''
python -m plotting.plot_gridworld_falls
'''

or 

'''
python -m plotting.plot_gridworld_train
'''

Since training an agent is quick in this environment, we do not provide pre-trained agents for the gridworld.

#Running experiments on MinAtar: example

To train a UA-DQN agent on Breakout, run the following command from this folder:

'''
python -m experiments.uadqn.train_minatar
'''

A log folder will automatically be created in the results folder, which at the end of training will contain the trained agent's network weights, log data, and plots of the agent's performance. 

Once several seeds have been saved, the training curves can be plotted by running:

'''
python -m plotting.plot_minatar
'''

# Results from pre-trained agents on MinAtar

In the results folder, we have included results from one of each of UADQN, QRDQN, DQN, and Bootstrapped DQN. Training curves from these agents can be plotted by running 

'''
python -m plotting.plot_minatar
'''
