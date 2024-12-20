if __name__ == '__main__':
    import sys
    from pathlib import Path
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import torch
import numpy as np
import os
from tqdm.auto import trange
from agents.networks.gru_rnn import GRU_RNN
from aux_funcs import zero_pad, make_path_if_not_exists, compressed_write
from datetime import datetime
import argparse
import multiprocessing as mp
import pickle
from copy import deepcopy as copy
import tracemalloc
from load_env import get_env_vars

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str)
parser.add_argument('--env', metavar='e', type=str, default='LOCAL')
args = parser.parse_args()

# GET MACHINE ENV VARS
env_vars = get_env_vars(args.env)

OUTPUT_BASE_DIR = os.path.join(env_vars['RESULTS_PATH'], 'line_attr_supervised')
OUTPUT_SAVE_RATE = 1

HIDDEN_SIZE = 4
INPUT_SIZE = 1
DEVICE = 'cuda'
VAR_NOISE = 1e-4
ACTIVITY_WEIGHT = 1e-7


def sample_from_markov_process(batch_size, t, transition_mat):
    # create (batch_size, t) mat
    markov_trajectories = np.zeros((batch_size, t)).astype(int)
    # sample for t successive states
    for k in range(t-1):
        state = markov_trajectories[:, k]
        p_transitions = transition_mat[state, :]
        markov_trajectories[:, k+1] = np.stack([
            np.random.choice(np.arange(transition_mat.shape[0]), p=p_transition)
            for p_transition in p_transitions
        ])

    return torch.from_numpy(markov_trajectories.reshape(batch_size, 1, t)).float()

if __name__ == '__main__':
    time_stamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    output_dir = os.path.join(OUTPUT_BASE_DIR, '_'.join([args.exp_title, time_stamp, f'var_noise_{VAR_NOISE}', f'activity_weight_{ACTIVITY_WEIGHT}']))
    outputs_output_dir = os.path.join(output_dir, 'outputs')
    hidden_state_output_dir = os.path.join(output_dir, 'hidden_states')
    make_path_if_not_exists(outputs_output_dir)
    make_path_if_not_exists(hidden_state_output_dir)

    t = 1000

    network = GRU_RNN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        device=DEVICE,
        var_noise=VAR_NOISE,
    )

    load_path = './results/line_attr_supervised/5_unit_line_2024-12-04_14_05_12_930912_var_noise_0.0001_activity_weight_1e-07/rnn_weights/009999.h5'
    network.load_state_dict(torch.load(load_path, weights_only=True))
    network.eval()

    losses = np.empty((OUTPUT_SAVE_RATE))

    with torch.no_grad():
        for k in trange(20):
            p_on = np.random.rand() * 0.05
            p_off = np.random.rand() * 0.05
            transition_mat = np.array([
                [1 - p_on, p_on],
                [p_off, 1 - p_off],
            ])

            inputs = sample_from_markov_process(100, t, transition_mat).detach().to(DEVICE)
            target_outputs = torch.sum(inputs, dim=2) / t
            outputs, activity = network(inputs)
            print('output:', outputs)
            print('target:', target_outputs.squeeze(1))
            loss = torch.pow(target_outputs.squeeze(1) - outputs, 2).mean() + ACTIVITY_WEIGHT * activity.pow(2).sum().mean()
            print(network.rnn.weight_hh.pow(2).sum().pow(0.5))
            print('loss:', loss)
            print('act_pen', ACTIVITY_WEIGHT * activity.pow(2).sum().mean())

            losses[k % OUTPUT_SAVE_RATE] = loss.clone().detach().cpu().numpy()

            padded_save_num = zero_pad(str(k), 6)
            np.save(
                os.path.join(outputs_output_dir, f'{padded_save_num}.npy'),
                np.stack([
                    outputs.detach().cpu().numpy(),
                    target_outputs.squeeze(1).detach().cpu().numpy(),
                ])
            )
            losses = np.empty((OUTPUT_SAVE_RATE))
            np.save(os.path.join(hidden_state_output_dir, f'{padded_save_num}.npy'), activity.clone().detach().cpu().numpy())

            network.reset_state()


