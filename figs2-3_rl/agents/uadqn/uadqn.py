"""
Implements our uncertainty-aware DQN, which uses QRDQN to estimate
aleatoric risk and Anchoring to estimate epistemic uncertainty.
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
from agents.common.utils import quantile_huber_loss


class UADQN:
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
    weight_scale: scale of prior neural network weights at initialization
    noise_scale: scale of aleatoric noise
    epistemic_factor: multiplier for epistemic uncertainty used for Thompson sampling
    aleatoric_factor: maulitplier for aleatoric uncertainty, used to adjust mean Q values
    update_target_frequency: Frequency at which target network is updated
    minibatch_size : Minibatch size.
    update_frequency : Number of environment steps taken between parameter update steps.
    learning_rate : Learning rate used for the Adam optimizer
    seed : The global seed to set.  None means randomly selected.
    adam_epsilon: Epsilon parameter for Adam optimizer
    biased_aleatoric: whether to use empirical std of quantiles as opposed to unbiased estimator

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
        noise_scale=0.1,
        epistemic_factor=1,
        aleatoric_factor=1,
        update_target_frequency=10000,
        minibatch_size=32,
        update_frequency=1,
        learning_rate=1e-3,
        seed=None,
        adam_epsilon=1e-8,
        biased_aleatoric=False,
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
        self.n_quantiles = n_quantiles
        self.kappa = kappa
        self.weight_scale = weight_scale
        self.noise_scale = noise_scale
        self.epistemic_factor = epistemic_factor,
        self.aleatoric_factor = aleatoric_factor,
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.seed = random.randint(0, 1e6) if seed is None else seed
        self.adam_epsilon = adam_epsilon
        self.biased_aleatoric = biased_aleatoric
        self.logging = logging
        self.log_folder_details = log_folder_details
        self.save_period = save_period
        self.render = render
        self.notes = notes

        # Set global seed before creating network
        set_global_seed(self.seed, self.env)

        # Initialize agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = None
        self.loss = quantile_huber_loss
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # Initialize main Q learning network
        n_outputs = self.env.action_space.n*self.n_quantiles
        self.network = network(self.env.observation_space, n_outputs).to(self.device)
        self.target_network = network(self.env.observation_space, n_outputs).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Initialize anchored networks
        self.posterior1 = network(self.env.observation_space, n_outputs, weight_scale=weight_scale).to(self.device)
        self.posterior2 = network(self.env.observation_space, n_outputs, weight_scale=weight_scale).to(self.device)
        self.anchor1 = [p.data.clone() for p in list(self.posterior1.parameters())]
        self.anchor2 = [p.data.clone() for p in list(self.posterior2.parameters())]

        # Initialize optimizer
        params = list(self.network.parameters()) + list(self.posterior1.parameters()) + list(self.posterior2.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate, eps=self.adam_epsilon)

        # Figure out what the scale of the prior is from empirical std of network weights
        with torch.no_grad():
            std_list = []
            for i, p in enumerate(self.posterior1.parameters()):
                std_list.append(torch.std(p))
        self.prior_scale = torch.stack(std_list).mean().item()

        # Parameters to save to log file
        self.train_parameters = {
                    'Notes': notes,
                    'env': str(env),
                    'network': str(self.network),
                    'n_quantiles': n_quantiles,
                    'replay_start_size': replay_start_size,
                    'replay_buffer_size': replay_buffer_size,
                    'gamma': gamma,
                    'update_target_frequency': update_target_frequency,
                    'minibatch_size': minibatch_size,
                    'learning_rate': learning_rate,
                    'update_frequency': update_frequency,
                    'weight_scale': weight_scale,
                    'noise_scale': noise_scale,
                    'epistemic_factor': epistemic_factor,
                    'aleatoric_factor': aleatoric_factor,
                    'biased_aleatoric': biased_aleatoric,
                    'adam_epsilon': adam_epsilon,
                    'seed': self.seed
                    }

    def learn(self, timesteps, verbose=False):

        self.non_greedy_actions = 0
        self.timestep = 0
        self.this_episode_time = 0
        self.n_events = 0  # Number of times an important event is flagged in the info

        self.train_parameters['train_steps'] = timesteps
        pprint.pprint(self.train_parameters)

        if self.logging:
            self.logger = Logger(self.log_folder_details, self.train_parameters)

        # Initialize the state
        state = torch.as_tensor(self.env.reset())
        score = 0
        t1 = time.time()

        for timestep in range(timesteps):

            is_training_ready = timestep >= self.replay_start_size

            if self.render:
                self.env.render()

            # Select action
            action = self.act(state.to(self.device).float(), is_training_ready=is_training_ready)

            # Perform action in environment
            state_next, reward, done, info = self.env.step(action)

            if (info == "The agent fell!") and self.logging:  # For gridworld experiments
                self.n_events += 1
                self.logger.add_scalar('Agent falls', self.n_events, timestep)

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            done = torch.as_tensor([done], dtype=torch.float)
            state_next = torch.as_tensor(state_next)
            self.replay_buffer.add(state, action, reward, state_next, done)

            score += reward.item()
            self.this_episode_time += 1

            if done:

                if verbose:
                    print("Timestep : {}, score : {}, Time : {} s".format(timestep, score, round(time.time() - t1, 3)))

                if self.logging:
                    self.logger.add_scalar('Episode_score', score, timestep)

                # Reinitialize the state
                state = torch.as_tensor(self.env.reset())
                score = 0

                if self.logging:
                    non_greedy_fraction = self.non_greedy_actions/self.this_episode_time
                    self.logger.add_scalar('Non Greedy Fraction', non_greedy_fraction, timestep)

                self.non_greedy_actions = 0
                self.this_episode_time = 0
                t1 = time.time()

            else:
                state = state_next

            if is_training_ready:

                # Update main network
                if timestep % self.update_frequency == 0:

                    # Sample batch of transitions
                    transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                    # Train on selected batch
                    loss, anchor_loss = self.train_step(transitions)

                    if self.logging and timesteps < 50000:
                        self.logger.add_scalar('Loss', loss, timestep)
                        self.logger.add_scalar('Anchor Loss', anchor_loss, timestep)

                # Periodically update target Q network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if (timestep+1) % self.save_period == 0:
                self.save(timestep=timestep+1)

            self.timestep += 1

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
            target = self.target_network(states_next.float())
            target = target.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)

        # Calculate max of target Q values
        best_action_idx = torch.mean(target, dim=2).max(1, True)[1].unsqueeze(2)
        q_value_target = target.gather(1, best_action_idx.repeat(1, 1, self.n_quantiles))

        # Calculate TD target
        rewards = rewards.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        dones = dones.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calculate Q value of actions played
        outputs = self.network(states.float())
        outputs = outputs.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)
        actions = actions.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        q_value = outputs.gather(1, actions)

        # TD loss for main network
        loss = self.loss(q_value.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)

        # Calculate predictions of posterior networks
        posterior1 = self.posterior1(states.float())
        posterior1 = posterior1.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)
        posterior1 = posterior1.gather(1, actions)

        posterior2 = self.posterior2(states.float())
        posterior2 = posterior2.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)
        posterior2 = posterior2.gather(1, actions)

        # Regression loss for the posterior networks
        loss_posterior1 = self.loss(posterior1.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss_posterior2 = self.loss(posterior2.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss += loss_posterior1 + loss_posterior2

        # Anchor loss for the posterior networks
        anchor_loss = self.calc_anchor_loss()        
        loss += anchor_loss

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), anchor_loss.mean().item()

    def calc_anchor_loss(self):
        """
        Returns loss from anchoring
        """

        diff1 = []
        for i, p in enumerate(self.posterior1.parameters()):
            diff1.append(torch.sum((p - self.anchor1[i])**2))
        diff1 = torch.stack(diff1).sum()

        diff2 = []
        for i, p in enumerate(self.posterior2.parameters()):
            diff2.append(torch.sum((p-self.anchor2[i])**2))
        diff2 = torch.stack(diff2).sum()

        diff = diff1 + diff2

        num_data = np.min([self.timestep, self.replay_buffer_size])
        anchor_loss = self.noise_scale**2*diff/(self.prior_scale**2*num_data)

        return anchor_loss

    @torch.no_grad()
    def get_q(self, state):
        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)
        action_means = torch.mean(net, dim=1)
        q = action_means
        return q

    @torch.no_grad()
    def act(self, state, is_training_ready=True):
        """
        Returns action to be performed using Thompson sampling
        with estimates provided by the two posterior networks
        """

        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)

        posterior1 = self.posterior1(state).view(self.env.action_space.n, self.n_quantiles)
        posterior2 = self.posterior2(state).view(self.env.action_space.n, self.n_quantiles)

        mean_action_values = torch.mean(net, dim=1)

        # Calculate aleatoric uncertainty
        if self.biased_aleatoric:
            uncertainties_aleatoric = torch.std(net, dim=1)
        else:
            covariance = torch.mean((posterior1-torch.mean(posterior1))*(posterior2-torch.mean(posterior2)), dim=1)
            uncertainties_aleatoric = torch.sqrt(F.relu(covariance))

        # Aleatoric-adjusted Q values
        aleatoric_factor = torch.FloatTensor(self.aleatoric_factor).to(self.device)
        adjusted_action_values = mean_action_values - aleatoric_factor*uncertainties_aleatoric

        # Calculate epistemic uncertainty
        uncertainties_epistemic = torch.mean((posterior1-posterior2)**2, dim=1)/2 + 1e-8
        epistemic_factor = torch.FloatTensor(self.epistemic_factor).to(self.device)**2
        uncertainties_cov = epistemic_factor*torch.diagflat(uncertainties_epistemic)

        # Draw samples using Thompson sampling
        epistemic_distrib = torch.distributions.multivariate_normal.MultivariateNormal
        samples = epistemic_distrib(adjusted_action_values, covariance_matrix=uncertainties_cov).sample()
        action = samples.argmax().item()

        #print(mean_action_values, torch.sqrt(uncertainties_epistemic), torch.sqrt(uncertainties_aleatoric))
        
        if self.logging and self.this_episode_time == 0:
            self.logger.add_scalar('Epistemic Uncertainty 0', torch.sqrt(uncertainties_epistemic)[0], self.timestep)
            self.logger.add_scalar('Epistemic Uncertainty 1', torch.sqrt(uncertainties_epistemic)[1], self.timestep)
            self.logger.add_scalar('Aleatoric Uncertainty 0', uncertainties_aleatoric[0], self.timestep)
            self.logger.add_scalar('Aleatoric Uncertainty 1', uncertainties_aleatoric[1], self.timestep)
            self.logger.add_scalar('Q0', mean_action_values[0], self.timestep)
            self.logger.add_scalar('Q1', mean_action_values[1], self.timestep)

        if action != mean_action_values.argmax().item():
            self.non_greedy_actions += 1
        
        return action

    @torch.no_grad()
    def predict(self, state):
        """
        Returns action with the highest Q-value
        """
        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)
        mean_action_values = torch.mean(net, dim=1)
        action = mean_action_values.argmax().item()

        return action

    def save(self, timestep=None):
        """
        Saves network weights
        """
        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
            filename_posterior1 = 'network_posterior1_' + str(timestep) + '.pth'
            filename_posterior2 = 'network_posterior2_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'
            filename_posterior1 = 'network_posterior1.pth'
            filename_posterior2 = 'network_posterior2.pth'

        save_path = self.logger.log_folder + '/' + filename
        save_path_posterior1 = self.logger.log_folder + '/' + filename_posterior1
        save_path_posterior2 = self.logger.log_folder + '/' + filename_posterior2

        torch.save(self.network.state_dict(), save_path)
        torch.save(self.posterior1.state_dict(), save_path_posterior1)
        torch.save(self.posterior2.state_dict(), save_path_posterior2)

    def load(self, path):
        """
        Loads network weights
        """
        self.network.load_state_dict(torch.load(path + 'network.pth', map_location='cpu'))
        self.posterior1.load_state_dict(torch.load(path + 'network_posterior1.pth', map_location='cpu'))
        self.posterior2.load_state_dict(torch.load(path + 'network_posterior2.pth', map_location='cpu'))
