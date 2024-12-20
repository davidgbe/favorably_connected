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
VAR_NOISE = 0 #1e-4
ACTIVITY_WEIGHT = 1e-7


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

    # print(network.rnn.bias_hh.size())
    # print(network.rnn.weight_ih.size())
    # print(network.rnn.bias_ih.size())

    gamma_w_hh_n = 1.5
    gamma_w_hh_z = 1
    gamma_w_ih_n = 1

    eig = np.ones((HIDDEN_SIZE, 1))
    eig = eig / np.linalg.norm(eig)
    w_ih_n = eig * gamma_w_ih_n
    w_hh_n = eig * eig.T * gamma_w_hh_n
    w_hh_z = np.outer(np.ones((HIDDEN_SIZE)), eig) * gamma_w_hh_z

    network.rnn.weight_hh.data = torch.from_numpy(np.zeros((3 * HIDDEN_SIZE, HIDDEN_SIZE))).float().to(DEVICE)
    network.rnn.weight_hh.data[2 * HIDDEN_SIZE : 3 * HIDDEN_SIZE, :] = torch.from_numpy(w_hh_n).float().to(DEVICE)
    network.rnn.weight_hh.data[1 * HIDDEN_SIZE : 2 * HIDDEN_SIZE, :] = torch.from_numpy(w_hh_z).float().to(DEVICE)

    network.rnn.weight_ih.data = torch.from_numpy(np.zeros((3 * HIDDEN_SIZE, 1))).float().to(DEVICE)
    network.rnn.weight_ih.data[HIDDEN_SIZE : 2 * HIDDEN_SIZE] = torch.from_numpy(2 * np.ones((HIDDEN_SIZE, 1))).float().to(DEVICE)
    network.rnn.weight_ih.data[2 * HIDDEN_SIZE : 3 * HIDDEN_SIZE, :] = torch.from_numpy(w_ih_n * gamma_w_ih_n).float().to(DEVICE)

    network.rnn.bias_hh.data = torch.from_numpy(np.zeros((3 * HIDDEN_SIZE,))).float().to(DEVICE)
    network.rnn.bias_ih.data = torch.from_numpy(np.zeros((3 * HIDDEN_SIZE,))).float().to(DEVICE)
    network.rnn.bias_ih.data[HIDDEN_SIZE:2 *HIDDEN_SIZE] = torch.from_numpy(4 * np.ones((HIDDEN_SIZE,))).float().to(DEVICE)
    # network.rnn.bias_ih.data[1 * HIDDEN_SIZE : 2 * HIDDEN_SIZE] = torch.from_numpy(1 * np.ones((HIDDEN_SIZE,))).float().to(DEVICE)

    network.output_arm.weight.data = torch.from_numpy(eig.T).float().to(DEVICE)
    network.output_arm.bias.data = torch.from_numpy(np.zeros(1,)).float().to(DEVICE)

    print(network.rnn.weight_hh.data)
    print(network.rnn.bias_hh.data)

    print(network.rnn.weight_ih.data)
    print(network.rnn.bias_ih.data)

    print(np.linalg.eig(network.rnn.weight_hh.data.clone().cpu()[2 * HIDDEN_SIZE : 3 * HIDDEN_SIZE, :]))


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
    
    print(w_hh_n)


