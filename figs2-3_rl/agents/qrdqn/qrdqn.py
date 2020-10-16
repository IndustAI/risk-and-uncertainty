"""
Implements QRDQN algorithm from Dabney et al,
"Distributional Reinforcement Learning with Quantile Regression"

Note: this class inherits quite a bit from DQN
"""

import torch
import torch.optim as optim
import numpy as np 

from agents.common.replay_buffer import ReplayBuffer
from agents.common.logger import Logger
from agents.common.utils import set_global_seed
from agents.dqn.dqn import DQN
from agents.common.utils import quantile_huber_loss


class QRDQN(DQN):
    """
    # Required parameters
    env : Environment to use.
    network : Choice of neural network.

    # Environment parameter
    gamma : Discount factor

    # Replay buffer
    replay_start_size : The capacity of the replay buffer at which training can begin.
    replay_buffer_size : Maximum buffer capacity.

    # QR-DQN parameters
    n_quantiles: Number of quantiles to estimate
    kappa: Smoothing parameter for the Huber loss
    update_target_frequency: Frequency at which target network is updated
    minibatch_size : Minibatch size.
    update_frequency : Number of environment steps taken between parameter update steps.
    learning_rate : Learning rate used for the Adam optimizer
    seed : The global seed to set.  None means randomly selected.
    adam_epsilon: Epsilon parameter for Adam optimizer

    # Exploration
    initial_exploration_rate : Inital exploration rate.
    final_exploration_rate : Final exploration rate.
    final_exploration_step : Timestep at which the final exploration rate is reached.

    # Logging and Saving
    logging : Whether to create logs when training
    log_folder_details : Additional information to put into the name of the log folder
    save_period : Periodicity with which the network weights are checkpointed
    notes : Notes to add to the log folder

    # Rendering
    render : Whether to render the environment during training. This slows down training.
    """

    def __init__(
        self,
        env,
        network,
        gamma=0.99,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        n_quantiles=50,
        kappa=1,
        weight_scale=3,
        update_target_frequency=10000,
        minibatch_size=32,
        update_frequency=1,
        learning_rate=1e-3,
        seed=None,
        adam_epsilon=1e-8,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        final_exploration_step=1000000,
        logging=False,
        log_folder_details=None,
        save_period=250000,
        notes=None,
        render=False,
    ):

        super().__init__(
                    env, network, gamma=gamma, replay_start_size=replay_start_size,
                    replay_buffer_size=replay_buffer_size, update_target_frequency=update_target_frequency,
                    minibatch_size=minibatch_size, update_frequency=update_frequency,
                    learning_rate=learning_rate, seed=seed, adam_epsilon=adam_epsilon,
                    initial_exploration_rate=initial_exploration_rate, final_exploration_rate=final_exploration_rate,
                    final_exploration_step=final_exploration_step, logging=logging,
                    log_folder_details=log_folder_details, save_period=save_period, notes=notes,
                    render=render, loss="huber"
                    )

        # Agent parameters
        self.n_quantiles = n_quantiles
        self.kappa = kappa

        # Initialize agent
        self.network = network(self.env.observation_space, self.env.action_space.n*self.n_quantiles, weight_scale=weight_scale).to(self.device)
        self.target_network = network(self.env.observation_space, self.env.action_space.n*self.n_quantiles).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)
        self.loss = quantile_huber_loss

        # Parameters to save to log file
        self.train_parameters = {'Notes': notes,
                'env': str(env),
                'network': str(self.network),
                'n_quantiles': n_quantiles,
                'kappa': kappa,
                'replay_start_size': replay_start_size,
                'replay_buffer_size': replay_buffer_size,
                'weight_scale': weight_scale,
                'gamma': gamma,
                'update_target_frequency': update_target_frequency,
                'weight_scale': self.network.weight_scale,
                'minibatch_size': minibatch_size,
                'learning_rate': learning_rate,
                'update_frequency': update_frequency,
                'initial_exploration_rate': initial_exploration_rate,
                'final_exploration_rate': final_exploration_rate,
                'final_exploration_step': final_exploration_step,
                'adam_epsilon': adam_epsilon,
                'seed': self.seed}

    def train_step(self, transitions):
        """
        Performs gradient descent step on a batch of transitions
        """

        states, actions, rewards, states_next, dones = transitions

        # Calculate target Q
        with torch.no_grad():
            target_outputs = self.target_network(states_next.float()).view(
                self.minibatch_size,
                self.env.action_space.n,
                self.n_quantiles
                )
            best_action_idx = torch.mean(target_outputs, dim=2).max(1, True)[1].unsqueeze(2)
            q_value_target = target_outputs.gather(1, best_action_idx.repeat(1, 1, self.n_quantiles))

        # Calculate TD target
        rewards = rewards.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        dones = dones.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calculate Q value of actions played
        outputs = self.network(states.float()).view(
            self.minibatch_size,
            self.env.action_space.n,
            self.n_quantiles
            )
        actions = actions.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        q_value = outputs.gather(1, actions)

        loss = self.loss(q_value.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def get_max_q(self, state):
        """
        Returns largest Q value at the state
        """

        net = self.network(state)
        action_means = net.view(self.env.action_space.n, self.n_quantiles).mean(dim=1)
        max_q = action_means.max().item()
        return max_q

    @torch.no_grad()
    def predict(self, state):
        """
        Returns action with the highest Q-value
        """
        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)
        mean_action_values = torch.mean(net, dim=1)

        uncertainties_aleatoric = torch.std(net, dim=1)

        if self.logging and self.this_episode_time == 0:
            self.logger.add_scalar('Aleatoric Uncertainty 0', uncertainties_aleatoric[0], self.timestep)
            self.logger.add_scalar('Aleatoric Uncertainty 1', uncertainties_aleatoric[1], self.timestep)
            self.logger.add_scalar('Q0', mean_action_values[0], self.timestep)
            self.logger.add_scalar('Q1', mean_action_values[1], self.timestep)

        action = mean_action_values.argmax().item()
        return action
