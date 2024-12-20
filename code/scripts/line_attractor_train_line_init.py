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
ACTIVITY_WEIGHT = 0

T = 500
DECODING_PERIOD = 200
BATCH_SIZE = 100

ATTR_POOL_E_SIZE = 16
ATTR_POOL_I_SIZE = 4
ATTR_POOL_W_EE = 1 / np.sqrt(HIDDEN_SIZE)
ATTR_POOL_W_EI = 1 / np.sqrt(HIDDEN_SIZE)
ATTR_POOL_W_IE = -1 / np.sqrt(HIDDEN_SIZE)


def sample_random_walks(batch_size, t, input_len, p_vec):
    # create (batch_size, t) mat
    trajectories = np.zeros((batch_size, t))
    p = p_vec.reshape(batch_size, 1) * np.ones((batch_size, input_len))
    trajectories[:, :input_len] = np.where(np.random.rand(batch_size, input_len) < p, 1, 0)
    return torch.from_numpy(trajectories.reshape(batch_size, 1, t)).float()

if __name__ == '__main__':
    time_stamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    output_dir = os.path.join(OUTPUT_BASE_DIR, '_'.join([args.exp_title, time_stamp, f'var_noise_{VAR_NOISE}', f'activity_weight_{ACTIVITY_WEIGHT}']))
    loss_output_dir = os.path.join(output_dir, 'losses')
    weights_output_dir = os.path.join(output_dir, 'rnn_weights')
    make_path_if_not_exists(loss_output_dir)
    make_path_if_not_exists(weights_output_dir)

    network = GRU_RNN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        device=DEVICE,
        var_noise=VAR_NOISE,
    )

    w_line_attr = np.zeros((ATTR_POOL_E_SIZE + ATTR_POOL_I_SIZE, ATTR_POOL_E_SIZE + ATTR_POOL_I_SIZE))

    x = np.arange(ATTR_POOL_E_SIZE) / ATTR_POOL_E_SIZE
    connectivity_scale = 0.075
    exp_ring_connectivity = 4 * ATTR_POOL_W_EE * (np.exp(-x/connectivity_scale) + np.exp((x-1)/connectivity_scale))
    
    for r_idx in np.arange(ATTR_POOL_E_SIZE):
        w_line_attr[r_idx:ATTR_POOL_E_SIZE, r_idx] = exp_ring_connectivity[:(ATTR_POOL_E_SIZE - r_idx)]
        w_line_attr[0:r_idx, r_idx] = exp_ring_connectivity[(ATTR_POOL_E_SIZE - r_idx):]
        
    w_line_attr[-ATTR_POOL_I_SIZE:, :ATTR_POOL_E_SIZE] = np.random.normal(size=(ATTR_POOL_I_SIZE, ATTR_POOL_E_SIZE), loc=ATTR_POOL_W_EI, scale=0.1 * ATTR_POOL_W_EI)
    w_line_attr[:ATTR_POOL_E_SIZE, -ATTR_POOL_I_SIZE:] = np.random.normal(size=(ATTR_POOL_E_SIZE, ATTR_POOL_I_SIZE), loc=ATTR_POOL_W_IE, scale=0.1 * np.abs(ATTR_POOL_W_IE))

    network.rnn.weight_hh.data[2 * HIDDEN_SIZE : 2 * HIDDEN_SIZE + ATTR_POOL_E_SIZE + ATTR_POOL_I_SIZE, : ATTR_POOL_E_SIZE + ATTR_POOL_I_SIZE] = torch.from_numpy(w_line_attr).float().to(DEVICE)

    scale = 10
    fig, axs = plt.subplots(1, 3, figsize=(4 * scale, 1 * scale))

    weight_hh = network.rnn.weight_hh.data.clone().detach().cpu().numpy()
    m = np.max(np.abs(weight_hh))
    for k in range(3):
        cbar = axs[k].matshow(weight_hh[k * HIDDEN_SIZE:(k+1) * HIDDEN_SIZE, :], vmin=-m, vmax=m, cmap='bwr')
        plt.colorbar(cbar)
    fig.savefig(os.path.join(output_dir, f'weights.png'))


    optimizer = torch.optim.RMSprop(network.parameters(), lr=LEARNING_RATE)
    losses = np.empty((OUTPUT_SAVE_RATE))

    for k in trange(200000):
        p_on = np.random.rand((BATCH_SIZE))
    
        inputs = sample_random_walks(BATCH_SIZE, T, T - DECODING_PERIOD, p_on).detach().to(DEVICE)
        target_outputs = torch.sum(inputs, dim=2) / T
        optimizer.zero_grad()
        outputs, activity = network(inputs, output_steps=DECODING_PERIOD)
        print(outputs)
        indices = torch.from_numpy(np.arange(BATCH_SIZE)).to(DEVICE)
        rand_times = torch.from_numpy(np.random.rand(BATCH_SIZE) * DECODING_PERIOD).int().to(DEVICE)
        sampled_outputs = outputs[indices, rand_times]

        print('output:', sampled_outputs)
        print('target:', target_outputs.squeeze(1))
        loss = torch.pow(target_outputs.squeeze(1) - sampled_outputs, 2).mean() + ACTIVITY_WEIGHT * activity.pow(2).sum().mean()
        print(network.rnn.weight_hh.pow(2).sum().pow(0.5))
        print('loss:', loss)
        print('act_pen', ACTIVITY_WEIGHT * activity.pow(2).sum().mean())

        losses[k % OUTPUT_SAVE_RATE] = loss.clone().detach().cpu().numpy()

        if k % OUTPUT_SAVE_RATE == OUTPUT_SAVE_RATE - 1:
            padded_save_num = zero_pad(str(k), 6)
            np.save(os.path.join(loss_output_dir, f'{padded_save_num}.npy'), losses)
            losses = np.empty((OUTPUT_SAVE_RATE))
            torch.save(network.state_dict(), os.path.join(weights_output_dir, f'{padded_save_num}.h5'))

        loss.backward()
        network.reset_state()
        optimizer.step()


