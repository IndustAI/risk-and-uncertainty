"""
Indust implementation of bootstrapped DQN
@author: William

This class is quite close to DQN, but there are enough differences
to warrant this being its own class.
"""

import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import pprint as pprint

from agents.common.replay_buffer import ReplayBuffer
from agents.common.logger import Logger
from agents.common.utils import set_global_seed


class BOOTSTRAPPED:
    """
    # Required parameters
    env : Environment to use.
    network : Choice of neural network.

    # Environment parameter
    gamma : Discount factor

    # Replay buffer
    replay_start_size : The capacity of the replay buffer at which training can begin.
    replay_buffer_size : Maximum buffer capacity.

    # Bootstrapped DQN parameters
    n_heads: Number of bootstrap heads
    update_target_frequency: Frequency at which target network is updated
    minibatch_size : Minibatch size.
    update_frequency : Number of environment steps taken between parameter update steps.
    learning_rate : Learning rate used for the Adam optimizer
    loss : Type of loss function to use. Can be "huber" or "mse"
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
        n_heads=10,
        update_target_frequency=10000,
        minibatch_size=32,
        update_frequency=1,
        learning_rate=1e-3,
        loss="huber",
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

        # Agent parameters
        self.env = env
        self.gamma = gamma
        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.n_heads = n_heads
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        if callable(loss):
            self.loss = loss
        else:
            try:
                self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[loss]
            except KeyError:
                raise ValueError("loss must be 'huber', 'mse' or a callable")
        self.seed = random.randint(0, 1e6) if seed is None else seed
        self.adam_epsilon = adam_epsilon
        self.initial_exploration_rate = initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_step = final_exploration_step
        self.logging = logging
        self.log_folder_details = log_folder_details
        self.save_period = save_period
        self.render = render
        self.notes = notes

        # Set global seed before creating network
        set_global_seed(self.seed, self.env)

        # Initialize agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.logger = None
        self.epsilon = self.initial_exploration_rate
        self.network = network(self.env.observation_space, self.env.action_space.n, self.n_heads).to(self.device)
        self.target_network = network(self.env.observation_space, self.env.action_space.n, self.n_heads).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)
        self.current_head = None
        self.timestep = 0

        # Parameters to save to log file
        self.train_parameters = {
                    'Notes': notes,
                    'env': str(env),
                    'network': str(self.network),
                    'replay_start_size': replay_start_size,
                    'replay_buffer_size': replay_buffer_size,
                    'gamma': gamma,
                    'n_heads': n_heads,
                    'update_target_frequency': update_target_frequency,
                    'minibatch_size': minibatch_size,
                    'learning_rate': learning_rate,
                    'update_frequency': update_frequency,
                    'initial_exploration_rate': initial_exploration_rate,
                    'final_exploration_rate': final_exploration_rate,
                    'weight_scale': self.network.weight_scale,
                    'final_exploration_step': final_exploration_step,
                    'adam_epsilon': adam_epsilon,
                    'loss': loss,
                    'seed': self.seed
                    }

    def learn(self, timesteps, verbose=False):

        self.current_head = np.random.randint(self.n_heads)

        self.train_parameters['train_steps'] = timesteps
        pprint.pprint(self.train_parameters)

        if self.logging:
            self.logger = Logger(self.log_folder_details, self.train_parameters)

        # Initialize the state
        state = torch.as_tensor(self.env.reset())
        score = 0
        t1 = time.time()

        for timestep in range(timesteps):

            self.timestep = timestep

            is_training_ready = timestep >= self.replay_start_size

            if self.render:
                self.env.render()

            # Select action
            action = self.act(state.to(self.device).float(), is_training_ready=is_training_ready)

            # Update epsilon
            self.update_epsilon(timestep)

            # Perform action in environment
            state_next, reward, done, _ = self.env.step(action)

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            done = torch.as_tensor([done], dtype=torch.float)
            state_next = torch.as_tensor(state_next)
            self.replay_buffer.add(state, action, reward, state_next, done)

            score += reward.item()

            if done:
                self.current_head = np.random.randint(self.n_heads)
                if self.logging:
                    self.logger.add_scalar('Acting_Head', self.current_head, self.timestep)

                # Reinitialize the state
                if verbose:
                    print("Timestep : {}, score : {}, Time : {} s".format(timestep, score, round(time.time() - t1, 3)))
                if self.logging:
                    self.logger.add_scalar('Episode_score', score, timestep)
                state = torch.as_tensor(self.env.reset())
                score = 0
                if self.logging:
                    self.logger.add_scalar('Q_at_start', self.get_max_q(state.to(self.device).float()), timestep)

                t1 = time.time()
            else:
                state = state_next

            if is_training_ready:

                # Update main network
                if timestep % self.update_frequency == 0:

                    # Sample batch of transitions
                    transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                    # Train on selected batch
                    loss = self.train_step(transitions)

                # Periodically update target Q network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if (timestep+1) % self.save_period == 0:
                self.save(timestep=timestep+1)

        if self.logging:
            self.logger.save()
            self.save()

        if self.render:
            self.env.close()

    def train_step(self, transitions):
        """
        Performs gradient descent step on a batch of transitions
        """

        states, actions, rewards, states_next, dones = transitions

        # Calculate target Q
        with torch.no_grad():

            # Target shape : batch x n_heads x n_actions
            targets = self.target_network(states_next.float())
            q_value_target = targets.max(2, True)[0]

        # Calculate TD target
        td_target = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * q_value_target

        # Calculate Q value of actions played
        output = self.network(states.float())
        q_value = output.gather(2, actions.unsqueeze(1).repeat(1,self.n_heads,1))

        loss = self.loss(q_value, td_target, reduction='mean')

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def get_max_q(self,state):
        """
        Returns largest Q value at the state
        """
        return self.network(state).max().item()

    def act(self, state, is_training_ready=True):
        """
        Returns action to be performed with an epsilon-greedy policy
        """

        if is_training_ready and random.uniform(0, 1) >= self.epsilon:
            # Action that maximizes Q
            action = self.predict(state)
        else:
            # Random action
            action = np.random.randint(0, self.env.action_space.n)

        return action

    def update_epsilon(self, timestep):
        """
        Update the exploration parameter
        """

        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
            timestep / self.final_exploration_step
        )
        self.epsilon = max(eps, self.final_exploration_rate)

    @torch.no_grad()
    def predict(self, state, train=True):
        """
        Returns action with the highest Q-value
        """

        if train:
            out = self.network(state).squeeze()
            action = out[self.current_head, :].argmax().item()
            """
            if self.logging:
                self.logger.add_scalar('Uncertainty', out.max(1,True)[0].std(), self.timestep)
            """
        else:
            out = self.network(state).squeeze()  # Shape B x n_heads x n_actions
            # The heads vote on the best action
            actions, count = torch.unique(out.argmax(1), return_counts=True)
            action = actions[count.argmax().item()].item()

        return action

    def save(self, timestep=None):
        """
        Saves network weights
        """

        if not self.logging:
            raise NotImplementedError('Cannot save without log folder.')

        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'

        save_path = self.logger.log_folder + '/' + filename

        torch.save(self.network.state_dict(), save_path)

    def load(self, path):
        """
        Loads network weights
        """

        self.network.load_state_dict(torch.load(path, map_location='cpu'))
