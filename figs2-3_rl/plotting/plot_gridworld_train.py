import os
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import pickle
import numpy as np
import matplotlib.pyplot as plt

TRAIN_STEPS = 10000
NUM_SEEDS = 10
PATH = 'results/GridExp/'

falls_qrdqn = []
falls_riskneutral = []
falls_riskaverse = []
falls_biasedrisk = []

for subdir, dirs, files in os.walk(PATH):

    if 'QRDQN' in subdir:

        log_data_filename = subdir + '/log_data.pkl'
        log_data = pickle.load(open(log_data_filename, 'rb'))
        score_data = np.array(log_data['Episode_score'])
        falls = score_data[:, 0]
        timesteps = score_data[:, 1]

        all_timesteps = np.arange(0, TRAIN_STEPS)
        spline = UnivariateSpline(timesteps, falls, k=1, s=0)
        falls_interpolated = spline(all_timesteps)

        falls_qrdqn.append(falls_interpolated)

    if 'UADQN' in subdir:

        info = eval(open(subdir + '/experimental-setup', 'r').read())

        if 'biased_aleatoric' in info.keys():

            log_data_filename = subdir + '/log_data.pkl'
            log_data = pickle.load(open(log_data_filename, 'rb'))
            score_data = np.array(log_data['Episode_score'])
            falls = score_data[:, 0]
            timesteps = score_data[:, 1]

            all_timesteps = np.arange(0, TRAIN_STEPS)
            spline = UnivariateSpline(timesteps, falls, k=1, s=0)
            falls_interpolated = spline(all_timesteps)

            falls_biasedrisk.append(falls_interpolated)

        else:
            if info['aletoric_factor'] == 0:

                log_data_filename = subdir + '/log_data.pkl'
                log_data = pickle.load(open(log_data_filename, 'rb'))
                score_data = np.array(log_data['Episode_score'])
                falls = score_data[:, 0]
                timesteps = score_data[:, 1]

                all_timesteps = np.arange(0, TRAIN_STEPS)
                spline = UnivariateSpline(timesteps, falls, k=1, s=0)
                falls_interpolated = spline(all_timesteps)

                falls_riskneutral.append(falls_interpolated)

            else:

                log_data_filename = subdir + '/log_data.pkl'
                log_data = pickle.load(open(log_data_filename, 'rb'))
                score_data = np.array(log_data['Episode_score'])
                falls = score_data[:, 0]
                timesteps = score_data[:, 1]

                all_timesteps = np.arange(0, TRAIN_STEPS)
                spline = UnivariateSpline(timesteps, falls, k=1, s=0)
                falls_interpolated = spline(all_timesteps)

                falls_riskaverse.append(falls_interpolated)


falls_qrdqn = np.array(falls_qrdqn)
n_qrdqn_seeds = falls_qrdqn.shape[0]
mean_qrdqn = savgol_filter(np.array(falls_qrdqn).mean(axis=0), 51, 3)
std_qrdqn = savgol_filter(np.array(falls_qrdqn).std(axis=0), 51, 3)

falls_riskneutral = np.array(falls_riskneutral)
n_riskneutral_seeds = falls_riskneutral.shape[0]
mean_riskneutral = savgol_filter(np.array(falls_riskneutral).mean(axis=0), 51, 3)
std_riskneutral = savgol_filter(np.array(falls_riskneutral).std(axis=0), 51, 3)

falls_riskaverse = np.array(falls_riskaverse)
n_riskaverse_seeds = falls_riskaverse.shape[0]
mean_riskaverse = savgol_filter(np.array(falls_riskaverse).mean(axis=0), 51, 3)
std_riskaverse = savgol_filter(np.array(falls_riskaverse).std(axis=0), 51, 3)

falls_biasedrisk = np.array(falls_biasedrisk)
n_biasedrisk_seeds = falls_biasedrisk.shape[0]
mean_biasedrisk = savgol_filter(np.array(falls_biasedrisk).mean(axis=0), 51, 3)
std_biasedrisk = savgol_filter(np.array(falls_biasedrisk).std(axis=0), 51, 3)

plt.figure(figsize=(8, 5))
plt.plot(mean_qrdqn, label='QRDQN', color='chocolate')
plt.plot(mean_riskneutral, label='UADQN-risk neutral', color='blue')
plt.plot(mean_biasedrisk, label='UADQN-risk averse 1', color='red')
plt.plot(mean_riskaverse, label='UADQN-risk averse 2', color='green')


plt.fill_between(
    all_timesteps,
    mean_qrdqn - 1.96 * std_qrdqn/np.sqrt(n_qrdqn_seeds),
    mean_qrdqn + 1.96 * std_qrdqn/np.sqrt(n_qrdqn_seeds),
    facecolor='orange',
    alpha=0.2)

plt.fill_between(
    all_timesteps,
    mean_riskneutral - 1.96 * std_riskneutral/np.sqrt(n_riskneutral_seeds),
    mean_riskneutral + 1.96 * std_riskneutral/np.sqrt(n_riskneutral_seeds),
    facecolor='blue',
    alpha=0.2)

plt.fill_between(
    all_timesteps,
    mean_riskaverse - 1.96 * std_riskaverse/np.sqrt(n_riskaverse_seeds),
    mean_riskaverse + 1.96 * std_riskaverse/np.sqrt(n_riskaverse_seeds),
    facecolor='green',
    alpha=0.2)

plt.fill_between(
    all_timesteps,
    mean_biasedrisk - 1.96 * std_biasedrisk/np.sqrt(n_biasedrisk_seeds),
    mean_biasedrisk + 1.96 * std_biasedrisk/np.sqrt(n_biasedrisk_seeds),
    facecolor='red',
    alpha=0.2)

plt.gcf().subplots_adjust(left=0.2, bottom=0.2)
plt.title('Training Curves', fontsize=20)
plt.xlabel('Step', fontsize=18)
plt.ylabel('Score', fontsize=18)
plt.ylim([-15, 6])
plt.xticks([0, 5000, 10000], fontsize=18)
plt.yticks([-10, -5, 0, 5], fontsize=18)
plt.legend(fontsize=16)
plt.show()
