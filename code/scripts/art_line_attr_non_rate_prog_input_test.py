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
OUTPUT_SAVE_RATE = 1

HIDDEN_SIZE = 32
INPUT_SIZE = 1
DEVICE = 'cuda'
VAR_NOISE = 0 #1e-4
ACTIVITY_WEIGHT = 1e-7

ATTR_POOL_E_SIZE = 16
ATTR_POOL_I_SIZE = 4
ATTR_POOL_W_EE = 1 / np.sqrt(HIDDEN_SIZE)
ATTR_POOL_W_EI = 1 / np.sqrt(HIDDEN_SIZE)
ATTR_POOL_W_IE = -1 / np.sqrt(HIDDEN_SIZE)


def gen_progressive_input(batch_size, t):
    # create (batch_size, t) mat
    markov_trajectories = np.zeros((batch_size, t)).astype(int)
    # sample for t successive states
    for k in range(batch_size):
        markov_trajectories[k, :int(k / batch_size * t)] = 1

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

    losses = np.empty((OUTPUT_SAVE_RATE))

    with torch.no_grad():
        for k in trange(20):
            inputs = gen_progressive_input(100, t).detach().to(DEVICE)
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


