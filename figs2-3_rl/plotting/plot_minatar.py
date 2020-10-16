import os
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

episode_length = 5000000
subsampling = 1000

scores_avg_ide = np.zeros(episode_length)
scores_avg_dqn = np.zeros(episode_length)

base_folder = 'results/'
label_by = ["Notes"]


fig, ax = plt.subplots(1, 5)

games = ['asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders']

for idx, criterion in enumerate(games):

    folders = ['all']

    if folders == ['all']:
        folders = [o for o in os.listdir(base_folder)]

    label_dict = {}
    for folder in folders:

        if not criterion or (criterion in folder):

            info = eval(open(base_folder + folder + '/experimental-setup', 'r').read())

            if label_by:
                label = ""
                for param in label_by:
                    label += str(info[param])
            else:
                label = folder

            log_data_filename = base_folder + folder + '/log_data.pkl'
            log_data = pickle.load(open(log_data_filename, 'rb'))
            score_data = np.array(log_data['Episode_score'])
            scores = score_data[:, 0]
            timesteps = score_data[:, 1]

            all_timesteps = np.arange(0,episode_length)
            spline = UnivariateSpline(timesteps, scores, k=1, s=0)
            scores = spline(all_timesteps)
            scores = pd.Series(scores)
            windows = scores.rolling(100000)
            scores = windows.mean()
            scores = np.where(scores < 0, 0, scores)
            scores = scores[0:episode_length-1:subsampling]

            if label in label_dict:
                label_dict[label].append(scores)
            else:
                label_dict[label] = [scores]
            

    sorted_label_dict = {}
    sorted_label_dict['UADQN'] = label_dict['UADQN']
    sorted_label_dict['QRDQN'] = label_dict['UADQN']
    sorted_label_dict['Bootstrapped'] = label_dict['Bootstrapped']
    sorted_label_dict['DQN'] = label_dict['DQN']

    for key in sorted_label_dict.keys():
        scores = np.array(label_dict[key])
        n_seeds = scores.shape[0]
        mean_scores = np.array(scores).mean(axis=0)
        ax[idx].plot(mean_scores, label=key)
        if n_seeds > 1:
            std_scores = np.array(scores).std(axis=0)
            ax[idx].fill_between(
                        np.arange(0,scores.shape[1]),
                        mean_scores - 1.96 * std_scores/np.sqrt(n_seeds),
                        mean_scores + 1.96 * std_scores/np.sqrt(n_seeds),
                        alpha=0.2)
        print(key, n_seeds)

    ax[idx].set_xticks([0, 5000])
    ax[idx].set_xticklabels([0, '5M'])
    ax[idx].tick_params(axis='both', which='major', labelsize=26)
    ax[idx].set_title(criterion, fontsize=32)

plt.show()