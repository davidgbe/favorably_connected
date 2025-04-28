import sys
from pathlib import Path

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import torch
import numpy as np
import os
import gymnasium as gym
from tqdm.auto import trange
from environments.treadmill_session import TreadmillSession
from environments.components.patch_type import PatchType
from environments.curriculum import Curriculum
from agents.networks.a2c_rnn_split import A2CRNN
from agents.a2c_recurrent_agent_split import A2CRecurrentAgent
from aux_funcs import zero_pad, make_path_if_not_exists, compressed_write, load_first_json, sample_truncated_exp
import optuna
from datetime import datetime
import argparse
import multiprocessing as mp
import pickle
from copy import deepcopy as copy
import tracemalloc
from load_env import get_env_vars
import matplotlib.pyplot as plt


# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str, default='run')
parser.add_argument('--noise_var', metavar='nv', type=float, default=1e-4)
parser.add_argument('--activity_reg', metavar='ar', type=float, default=1)
parser.add_argument('--curr_style', metavar='cs', type=str, default='FIXED')
parser.add_argument('--env', metavar='e', type=str, default='LOCAL')
parser.add_argument('--agent_type', metavar='at', type=str, default='split')
parser.add_argument('--connectivity', metavar='at', type=str, default='VANILLA')
parser.add_argument('--pipeline', metavar='p', type=int, default=0)
args = parser.parse_args()

curr_file_path = Path(__file__)

if bool(args.pipeline):
    if args.env == 'CODE_OCEAN':
        data, file_name = load_first_json(os.path.join(curr_file_path.parent.parent.parent, 'data'))
    else:
        data, file_name = load_first_json(os.path.join(curr_file_path.parent.parent.parent, 'results/rl_agent_outputs'))
    args.noise_var = data['noise_var']
    args.activity_reg = data['activity_reg']
    args.curr_style = data['curr_style']
    args.exp_title = file_name.replace('.json', '')

# GET MACHINE ENV VARS
env_vars = get_env_vars(args.env)

# ENVIRONEMENT PARAMS
PATCH_TYPES_PER_ENV = 3
OBS_SIZE = PATCH_TYPES_PER_ENV + 1
ACTION_SIZE = 2
DWELL_TIME_FOR_REWARD = 6
# for actual task, reward sites are 50 cm long and interreward sites are between 20 and 100, decay rate 0.05 (truncated exp.)
REWARD_SITE_LEN = 3
INTERREWARD_SITE_LEN_BOUNDS = [1, 6]
INTEREWARD_SITE_LEN_DECAY_RATE = 0.8
REWARD_DECAY_CONSTS = [0, 10, 30]
REWARD_PROB_PREFACTOR = 0.8
# for actual task, interpatch lengths are 200 to 600 cm, decay rate 0.01 
INTERPATCH_LEN_BOUNDS = [1, 12]
INTERPATCH_LEN_DECAY_RATE = 0.1
CURRICULUM_STYLE = args.curr_style
INPUT_NOISE_STD = 0.1

# AGENT PARAMS
HIDDEN_SIZE = 128
ATTR_POOL_SIZE = 15
CRITIC_WEIGHT = 0.07846647668470078
ENTROPY_WEIGHT = 1.0158892869509133e-06
GAMMA = 0.9867118269299845
LEARNING_RATE = 1e-4 #0.0006006712322528219

# TRAINING PARAMS
NUM_ENVS = 30
N_SESSIONS = 20000
N_UPDATES_PER_SESSION = 100
N_STEPS_PER_UPDATE = 200

# OTHER PARMS
DEVICE = 'cuda'
OUTPUT_STATE_SAVE_RATE = 1 # save one in 10 sessions
if args.env == 'CODE_OCEAN':
    OUTPUT_BASE_DIR = env_vars['RESULTS_PATH']
else:
    OUTPUT_BASE_DIR = os.path.join(env_vars['RESULTS_PATH'], 'rl_agent_outputs')


def interreward_site_transition():
    return int(sample_truncated_exp(
        size=1,
        bounds=INTERREWARD_SITE_LEN_BOUNDS,
        decay_rate=INTEREWARD_SITE_LEN_DECAY_RATE,
    )[0])


def interpatch_transition():
    return int(sample_truncated_exp(
        size=1,
        bounds=INTERPATCH_LEN_BOUNDS,
        decay_rate=INTERPATCH_LEN_DECAY_RATE,
    )[0])


def make_deterministic_treadmill_environment(env_idx):

    def make_env():
        np.random.seed(env_idx)

        print('Begin det. treadmill')

        patch_types = []
        for i in range(PATCH_TYPES_PER_ENV):
            def reward_func(site_idx):
                return 1
            patch_types.append(
                PatchType(
                    reward_site_len=REWARD_SITE_LEN,
                    interreward_site_len_func=interreward_site_transition,
                    reward_func=reward_func,
                    odor_num=i,
                    reward_func_param=0.0,
                )
            )

        transition_mat = 1/3 * np.ones((PATCH_TYPES_PER_ENV, PATCH_TYPES_PER_ENV))

        sesh = TreadmillSession(
            patch_types=patch_types,
            transition_mat=transition_mat,
            interpatch_len_func=interpatch_transition,
            dwell_time_for_reward=DWELL_TIME_FOR_REWARD,
            obs_size=PATCH_TYPES_PER_ENV + 1,
            verbosity=False,
        )


        return sesh

    return make_env


def make_stochastic_treadmill_environment(env_idx):

    def make_env():
        np.random.seed(env_idx + NUM_ENVS)
        
        decay_consts_for_reward_funcs = copy(REWARD_DECAY_CONSTS)
        if CURRICULUM_STYLE == 'MIXED':
            np.random.shuffle(decay_consts_for_reward_funcs)

        print('Begin stoch. treadmill')
        print(decay_consts_for_reward_funcs)

        patch_types = []
        for i in range(PATCH_TYPES_PER_ENV):
            decay_const_for_i = decay_consts_for_reward_funcs[i]
            active = (decay_const_for_i != 0)
            def reward_func(site_idx, decay_const_for_i=decay_const_for_i, active=active):
                c = REWARD_PROB_PREFACTOR * np.exp(-site_idx / decay_const_for_i) if decay_const_for_i > 0 else 0
                if np.random.rand() < c and active:
                    return 1
                else:
                    return 0
            patch_types.append(
                PatchType(
                    reward_site_len=REWARD_SITE_LEN,
                    interreward_site_len_func=interreward_site_transition,
                    reward_func=reward_func,
                    odor_num=i,
                    reward_func_param=(decay_consts_for_reward_funcs[i] if active else 0.0),
                )
            )

        transition_mat = 1/3 * np.ones((PATCH_TYPES_PER_ENV, PATCH_TYPES_PER_ENV))

        sesh = TreadmillSession(
            patch_types=patch_types,
            transition_mat=transition_mat,
            interpatch_len_func=interpatch_transition,
            dwell_time_for_reward=DWELL_TIME_FOR_REWARD,
            obs_size=PATCH_TYPES_PER_ENV + 1,
            verbosity=False,
        )

        return sesh

    return make_env



def objective(trial, var_noise, activity_weight):
    time_stamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    output_dir = os.path.join(OUTPUT_BASE_DIR, '_'.join([args.exp_title, time_stamp, f'var_noise_{var_noise}', f'activity_weight_{activity_weight}']))
    reward_rates_output_dir = os.path.join(output_dir, 'reward_rates')
    info_output_dir = os.path.join(output_dir, 'state')
    weights_output_dir = os.path.join(output_dir, 'rnn_weights')
    make_path_if_not_exists(reward_rates_output_dir)
    make_path_if_not_exists(info_output_dir)
    make_path_if_not_exists(weights_output_dir)

    if trial is not None:
        gamma = trial.suggest_float('gamma', 0.9, 1.0)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True)
        critic_weight = trial.suggest_float('critic_weight', 1e-4, 1e-1, log=True)
        entropy_weight = trial.suggest_float('entropy_weight', 1e-6, 1e-2, log=True)
    else:
        gamma = GAMMA
        learning_rate = LEARNING_RATE
        critic_weight = CRITIC_WEIGHT
        entropy_weight = ENTROPY_WEIGHT

    network = A2CRNN(
        input_size=OBS_SIZE + ACTION_SIZE + 1,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        device=DEVICE,
        var_noise=var_noise,
    )

    if args.connectivity == 'LINE':
        w_line_attr_pc = np.ones((ATTR_POOL_SIZE)) * (1.5 / np.sqrt(ATTR_POOL_SIZE))
        w_line_attr = np.outer(w_line_attr_pc, w_line_attr_pc)
        network.rnn.weight_hh.data[2 * HIDDEN_SIZE : 2 * HIDDEN_SIZE + ATTR_POOL_SIZE, : ATTR_POOL_SIZE] = torch.from_numpy(w_line_attr).float().to(DEVICE)

    if args.connectivity == 'LINE_PLUS_INPUT':
        w_line_attr_pc = np.ones((ATTR_POOL_SIZE)) * (1.5 / np.sqrt(ATTR_POOL_SIZE))
        w_line_attr = np.outer(w_line_attr_pc, w_line_attr_pc)
        network.rnn.weight_hh.data[2 * HIDDEN_SIZE : 2 * HIDDEN_SIZE + ATTR_POOL_SIZE, : ATTR_POOL_SIZE] = torch.from_numpy(w_line_attr).float().to(DEVICE)

        i = 2
        network.rnn.weight_ih.data[i * HIDDEN_SIZE : i * HIDDEN_SIZE + ATTR_POOL_SIZE, :6] = torch.zeros((ATTR_POOL_SIZE, 6)).float().to(DEVICE)
        network.rnn.weight_ih.data[i * HIDDEN_SIZE : i * HIDDEN_SIZE + ATTR_POOL_SIZE, 6] =  torch.from_numpy(w_line_attr_pc).float().to(DEVICE)

        i = 1
        network.rnn.weight_ih.data[i * HIDDEN_SIZE : i * HIDDEN_SIZE + ATTR_POOL_SIZE, 1:] = torch.zeros((ATTR_POOL_SIZE, 6)).float().to(DEVICE)
        network.rnn.weight_ih.data[i * HIDDEN_SIZE : i * HIDDEN_SIZE + ATTR_POOL_SIZE, 0] =  torch.from_numpy(w_line_attr_pc).float().to(DEVICE)

        scale = 10
        fig, axs = plt.subplots(1, 3, figsize=(4 * scale, 1 * scale))

        weight_hh = network.rnn.weight_hh.data.clone().detach().cpu().numpy()
        m = np.max(np.abs(weight_hh))
        for k in range(3):
            cbar = axs[k].matshow(weight_hh[k * HIDDEN_SIZE:(k+1) * HIDDEN_SIZE, :], vmin=-m, vmax=m, cmap='bwr')
            plt.colorbar(cbar)
        fig.savefig(os.path.join(output_dir, f'weights.png'))

        weight_ih = network.rnn.weight_ih.data.clone().detach().cpu().numpy()
        m = np.max(np.abs(weight_ih))
        for k in range(3):
            cbar = axs[k].matshow(weight_ih[k * HIDDEN_SIZE:(k+1) * HIDDEN_SIZE, :], vmin=-m, vmax=m, cmap='bwr')
            plt.colorbar(cbar)
        fig.savefig(os.path.join(output_dir, f'input_weights.png'))


    optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate)

    curricum = Curriculum(
        curriculum_step_starts=[0, 20],
        curriculum_step_env_funcs=[
            make_deterministic_treadmill_environment,
            make_stochastic_treadmill_environment,
        ],
    )

    env_seeds = np.arange(NUM_ENVS)
    save_num = 0
    last_snapshot = None

    for session_num in trange(N_SESSIONS, desc='Sessions'):

        agent = A2CRecurrentAgent(
            network,
            action_space_dims=ACTION_SIZE,
            n_envs=NUM_ENVS,
            device=DEVICE,
            critic_weight=critic_weight, # changed for Optuna
            entropy_weight=entropy_weight, # changed for Optuna
            gamma=gamma, # changed for Optuna
            learning_rate=learning_rate, # changed for Optuna
            activity_weight=activity_weight,
            optimizer=optimizer,
            input_noise_std=INPUT_NOISE_STD,
        )

        total_losses = np.empty((N_UPDATES_PER_SESSION))
        actor_losses = np.empty((N_UPDATES_PER_SESSION))
        critic_losses = np.empty((N_UPDATES_PER_SESSION))
        entropy_losses = np.empty((N_UPDATES_PER_SESSION))
        avg_rewards_per_update = np.empty((NUM_ENVS, N_UPDATES_PER_SESSION))
        all_info = []
        envs = curricum.get_envs_for_step(env_seeds)
        # at the start of training reset all envs to get an initial state
        # play n steps in our parallel environments to collect data
        for update_num in trange(N_UPDATES_PER_SESSION, desc='Updates in session'):
            if update_num == 0:
                obs, info = envs.reset()

            total_rewards = np.empty((NUM_ENVS, N_STEPS_PER_UPDATE))
            for step in range(N_STEPS_PER_UPDATE):
                action = agent.sample_action(obs).clone().detach().cpu().numpy()
                obs, reward, terminated, truncated, info = envs.step(action)
                agent.append_reward(reward.astype('float32'))
                total_rewards[:, step] = reward
                all_info.append(info)

            avg_rewards_per_update[:, update_num] = np.mean(total_rewards, axis=1)
            total_loss, actor_loss, critic_loss, entropy_loss = agent.get_losses()
            agent.update(total_loss)
            hidden_states, critic_hidden_states = agent.reset_state()
            agent.set_state(hidden_states, critic_hidden_states)

            total_losses[update_num] = total_loss.detach().cpu().numpy()
            actor_losses[update_num] = actor_loss.detach().cpu().numpy()
            critic_losses[update_num] = critic_loss.detach().cpu().numpy()
            entropy_losses[update_num] = entropy_loss.detach().cpu().numpy()

        padded_save_num = zero_pad(str(save_num), 5)
        np.save(os.path.join(reward_rates_output_dir, f'{padded_save_num}.npy'), avg_rewards_per_update)
        print('Avg reward for session:', np.mean(avg_rewards_per_update))
        if session_num % OUTPUT_STATE_SAVE_RATE == 0 and session_num > 0:
            try:
                compressed_write(all_info, os.path.join(info_output_dir, f'{padded_save_num}.pkl'))
                torch.save(
                    {
                        'network_state_dict': network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    },
                    os.path.join(weights_output_dir, f'{padded_save_num}.pth')
                )
            except MemoryError as me:
                print('Pickle dump caused memory crash')
                print(me)
                pass
        save_num += 1
        agent.reset_state()

    final_value = avg_rewards_per_update.mean()
    print('final_reward_total', final_value)
    print('gamma', gamma)
    print('learning_rate', learning_rate)
    print('critic_weight', critic_weight)
    print('entropy_weight', entropy_weight)

    return final_value

if __name__ == "__main__":
    print(objective(None, args.noise_var, args.activity_reg))