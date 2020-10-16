# Risk and Uncertainty in Deep Reinforcement Learning

Code associated with our paper "Estimating Risk and Uncertainty in Deep Reinforcement Learning", which was presented at the 2020 Workshop on Uncertainty and Robustness in Deep Learning at ICML 2020. You can find our paper <ahref="https://arxiv.org/abs/1905.09638">here</a>.

# Requirements

Requirements can be found in the requirements.txt file. 

In addition to these requirements, the MinAtar package should also be installed to reproduce the experiments presented in figure 4. See <a href="https://github.com/kenjyoung/MinAtar">this github repo</a> for installation instructions.

# Structure

Each folder is self-contained, and contains the code to reproduce the results presented in specific parts of our paper (figures 1-4 in the main text, and all figures in the supplementary materials). Each folder contains its own README with specific instructions for training and evaluation.

In all cases, we have included raw results from one pre-trained agent per algorithm, as well as code to plot its training curve. We have also included network weights for one seed of all agents for all games on the MinAtar suite.
