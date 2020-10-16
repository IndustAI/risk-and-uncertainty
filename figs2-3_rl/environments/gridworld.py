"""
Windy gridworld environment with a cliff
"""
import numpy as np

class Gridworld():

    def __init__(self,
                 world_height=2,
                 world_width=3,
                 wind_probability=0.2,
                 fall_reward=-5,
                 success_reward=10,
                 step_reward=-1,
                 max_timesteps=15
                 ):

        class action_space():
            """
            For discrete actions, action_space.n is the number of actions.
            2 actions by default, as in Cartpole.
            """
            def __init__(self): 
                self.n = 4

        class observation_space():
            """
            observation_space.shape is the shape of observations. 
            shape=[4] by default, as in Cartpole.
            """
            def __init__(self, width, height): 
                self.shape = [width*height+1]

        self.action_space = action_space()
        self.observation_space = observation_space(world_height, world_width)
        self.world_width = world_width
        self.world_height = world_height
        self.wind_probability = wind_probability
        self.fall_reward = fall_reward
        self.success_reward = success_reward
        self.step_reward = step_reward
        self.max_timesteps = max_timesteps

        self.position = [0, world_height-1]
        self.timestep = 0

    def step(self, action):
        """
        Requires: selected action
        Returns: observation, reward, done (boolean), and optional Notes
        """
        done = False
        has_fallen = False
        info = None

        reward = self.step_reward

        if action == 0:  # go up
            if self.position[1] != 0:
                self.position[1] -= 1

        if action == 1:  # go right
            if self.position[0] != self.world_width-1:
                self.position[0] += 1

        if action == 2:  # go down
            if self.position[1] != self.world_height-1:
                self.position[1] += 1
            else:  # If on the edge but not at start or finish, then fall
                if (self.position[0] != 0) and (self.position[0] != self.world_width-1):
                    has_fallen = True

        if action == 3:  # go left
            if self.position[0] != 0:
                self.position[0] -= 1

        # The wind can blow agent off cliff
        if (np.random.sample() < self.wind_probability) and (self.position[1] == self.world_height-1):
            if (self.position[0] != 0) and (self.position[0] != self.world_width-1):
                has_fallen = True

        if self.position == [self.world_width-1, self.world_height-1]:
            done = True
            reward += self.success_reward

        if has_fallen:  # If agent falls
            done = True
            reward += self.fall_reward
            info = "The agent fell!"

        self.timestep += 1
        if (self.timestep == self.max_timesteps) and not done:
            done = True

        return self.return_obs(), reward, done, info

    def reset(self):
        """
        Resets the environment and returns the first observation
        """
        self.position = [0, self.world_height-1]
        self.timestep = 0
        return self.return_obs()

    def return_obs(self):
        """
        Returns one-hot vector of agent position
        """
        grid_state = np.zeros((self.world_width, self.world_height))
        grid_state[self.position[0], self.position[1]] = 1
        grid_state = grid_state.flatten()
        time_state = np.array([self.timestep])
        obs = np.concatenate([grid_state, time_state])
        return obs

    def seed(self, seed):
        """
        Set random seed of the environment
        """

    def render(self):
        """
        Can be used to optionally render the environment
        """
