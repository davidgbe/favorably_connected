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
import matplotlib as mpl
import matplotlib.pyplot as plt

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str)
parser.add_argument('--env', metavar='e', type=str, default='LOCAL')
args = parser.parse_args()

# GET MACHINE ENV VARS
env_vars = get_env_vars(args.env)

OUTPUT_BASE_DIR = os.path.join(env_vars['RESULTS_PATH'], 'line_attr_supervised')
OUTPUT_SAVE_RATE = 1000

HIDDEN_SIZE = 32
INPUT_SIZE = 1
DEVICE = 'cuda'
LEARNING_RATE = 2e-4
VAR_NOISE = 0.5e-4
ACTIVITY_WEIGHT = 1e-7

T = 500
DECODING_PERIOD = 200
BATCH_SIZE = 100


def sample_random_walks(batch_size, t, input_len, p_vec):
    # create (batch_size, t) mat
    trajectories = np.zeros((batch_size, t))
    p = p_vec.reshape(batch_size, 1) * np.ones((batch_size, input_len))
    trajectories[:, :input_len] = np.where(np.random.rand(batch_size, input_len) < p, 1, -1)
    return torch.from_numpy(trajectories.reshape(batch_size, 1, t)).float()

if __name__ == '__main__':
    time_stamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    output_dir = os.path.join(OUTPUT_BASE_DIR, '_'.join([args.exp_title, time_stamp, f'var_noise_{VAR_NOISE}', f'activity_weight_{ACTIVITY_WEIGHT}']))
    outputs_output_dir = os.path.join(output_dir, 'outputs')
    hidden_state_output_dir = os.path.join(output_dir, 'hidden_states')
    weights_output_dir = os.path.join(output_dir, 'rnn_weights')
    make_path_if_not_exists(outputs_output_dir)
    make_path_if_not_exists(hidden_state_output_dir)
    make_path_if_not_exists(weights_output_dir)

    network = GRU_RNN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        device=DEVICE,
        var_noise=VAR_NOISE,
    )

    load_path = './results/line_attr_supervised/ramping_la_plus_min_150_2025-01-24_13_26_24_985440_var_noise_5e-05_activity_weight_1e-07/rnn_weights/003999.h5'
    network.load_state_dict(torch.load(load_path, weights_only=True))

    # v = np.real(np.linalg.eig(network.rnn.weight_hh.data[2 * HIDDEN_SIZE:3 *HIDDEN_SIZE, :].cpu().numpy()).eigenvectors[:, 0])

    # w_reset = 0.4 * np.outer(np.where(np.random.rand(HIDDEN_SIZE) > 0.5, 1, -1), v)
    # network.rnn.weight_hh.data[:HIDDEN_SIZE, :] = torch.from_numpy(w_reset).float().to(DEVICE)
    # network.rnn.bias_hh.data[:HIDDEN_SIZE] = torch.zeros_like(network.rnn.bias_hh.data[:HIDDEN_SIZE])
    network.eval()

    torch.save(network.state_dict(), os.path.join(weights_output_dir, f'weights.h5'))

    scale = 10
    fig, axs = plt.subplots(1, 3, figsize=(4 * scale, 1 * scale))

    weight_hh = network.rnn.weight_hh.data.clone().detach().cpu().numpy()
    m = np.max(np.abs(weight_hh))
    for k in range(3):
        cbar = axs[k].matshow(weight_hh[k * HIDDEN_SIZE:(k+1) * HIDDEN_SIZE, :], vmin=-m, vmax=m, cmap='bwr')
        plt.colorbar(cbar)
    fig.savefig(os.path.join(output_dir, f'weights.png'))


    losses = np.empty((OUTPUT_SAVE_RATE))

    with torch.no_grad():
        for k in trange(20):
            p_on = np.random.rand((BATCH_SIZE))
        
            inputs = sample_random_walks(BATCH_SIZE, T, T - DECODING_PERIOD, p_on).detach().to(DEVICE)
            target_outputs = torch.sum(inputs, dim=2) / (T - DECODING_PERIOD)

            outputs, activity = network(inputs, output_steps=DECODING_PERIOD)
            indices = torch.from_numpy(np.arange(BATCH_SIZE)).to(DEVICE)
            rand_times = torch.from_numpy(np.random.rand(BATCH_SIZE) * DECODING_PERIOD).int().to(DEVICE)
            sampled_outputs = outputs[indices, rand_times]

            print('output:', sampled_outputs)
            print('target:', target_outputs.squeeze(1))
            loss = torch.pow(target_outputs.squeeze(1) - sampled_outputs, 2).mean() + ACTIVITY_WEIGHT * activity.pow(2).sum().mean()
            print('loss:', loss)

            losses[k % OUTPUT_SAVE_RATE] = loss.clone().detach().cpu().numpy()

            padded_save_num = zero_pad(str(k), 6)
            np.save(
                os.path.join(outputs_output_dir, f'{padded_save_num}.npy'),
                np.stack([
                    sampled_outputs.detach().cpu().numpy(),
                    target_outputs.squeeze(1).detach().cpu().numpy(),
                ])
            )
            losses = np.empty((OUTPUT_SAVE_RATE))
            np.save(os.path.join(hidden_state_output_dir, f'{padded_save_num}.npy'), activity.clone().detach().cpu().numpy())

            network.reset_state()


