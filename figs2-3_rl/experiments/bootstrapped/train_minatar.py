from agents.common.networks.cnn_minatar import CNNMinAtar_k_head
from agents.bootstrapped.bootstrapped import BOOTSTRAPPED
from environments.minatar_wrappers import MinatarWrapper
from minatar import Environment
import argparse
import random

parser = argparse.ArgumentParser(description="Select MinAtar game")
parser.add_argument("--game", help="Atari game on which to train", default="breakout")
parser.add_argument("--nb_steps", help="Number of training steps", type=int, default=5000000)
parser.add_argument("--n_seeds", help="Number of training seeds", type=int, default=1)
params = parser.parse_args()

if params.game == "all":
    games = ["seaquest", "asterix", "breakout", "freeway", "space_invaders"]
else:
    games = [params.game]

for i in range(params.n_seeds):

    for game in games:

        seed = random.randint(0, 1e6)

        notes = "Bootstrapped"
        env = Environment(game, random_seed=seed)
        env = MinatarWrapper(env)
        nb_steps = params.nb_steps

        agent = BOOTSTRAPPED(env,
                CNNMinAtar_k_head,
                replay_start_size=5000,
                replay_buffer_size=100000,
                gamma=0.99,
                update_target_frequency=1000,
                minibatch_size=32,
                n_heads = 10,
                learning_rate=1e-4,
                update_frequency=1,
                initial_exploration_rate=1,
                final_exploration_rate=0.03,
                final_exploration_step=100000,
                adam_epsilon=1e-8,
                logging=True,
                log_folder_details=game+'-Bootstrapped',
                render=False,
                loss="huber",
                notes=None,
                save_period=1e7,
                seed=seed)

        agent.learn(timesteps=nb_steps, verbose=False)
