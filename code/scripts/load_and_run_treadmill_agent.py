if __name__ == '__main__':
    import sys
    from pathlib import Path
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import torch
import numpy as np
import os
import gymnasium as gym
from tqdm.auto import trange
from environments.treadmill_session import TreadmillSession
from environments.components.patch import Patch
from environments.curriculum import Curriculum
from agents.networks.a2c_rnn import A2CRNN
from agents.a2c_recurrent_agent import A2CRecurrentAgent
from aux_funcs import zero_pad, make_path_if_not_exists, compressed_write
import optuna
from datetime import datetime
import argparse
import multiprocessing as mp
import pickle
from copy import deepcopy as copy
import tracemalloc

# tracemalloc.start()


# ENVIRONEMENT PARAMS
PATCH_TYPES_PER_ENV = 3
OBS_SIZE = PATCH_TYPES_PER_ENV + 2
ACTION_SIZE = 2
DWELL_TIME_FOR_REWARD = 6
SPATIAL_BUFFER_FOR_VISUAL_CUES = 1.5
MAX_REWARD_SITE_LEN = 2
MIN_REWARD_SITE_LEN = 2
MAX_N_REWARD_SITES_PER_PATCH = 16
MIN_N_REWARD_SITES_PER_PATCH = 16
INTERREWARD_SITE_LEN_MEAN = 2
REWARD_DECAY_CONSTS = [0, 10, 30]
REWARD_PROB_PREFACTOR = 0.8
INTERPATCH_LEN = 6

# AGENT PARAMS
HIDDEN_SIZE = 128
CRITIC_WEIGHT = 0.07846647668470078
ENTROPY_WEIGHT = 1.0158892869509133e-06
GAMMA = 0.9867118269299845
LEARNING_RATE = 1e-4 #0.0006006712322528219

# TRAINING PARAMS
NUM_ENVS = 30
N_SESSIONS = 8
N_UPDATES_PER_SESSION = 100
N_STEPS_PER_UPDATE = 200

# OTHER PARMS
DEVICE = 'cuda'
OUTPUT_STATE_SAVE_RATE = 1 # save every session
OUTPUT_BASE_DIR = './data/rl_agent_outputs'

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str)
parser.add_argument('--load_path', metavar='lp', type=str)
args = parser.parse_args()



def make_deterministic_treadmill_environment(env_idx):

    def make_env():
        np.random.seed(env_idx)
        
        n_reward_sites_for_patches = np.random.randint(MIN_N_REWARD_SITES_PER_PATCH, high=MAX_N_REWARD_SITES_PER_PATCH + 1, size=(PATCH_TYPES_PER_ENV,))
        reward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_SITE_LEN - MIN_REWARD_SITE_LEN) + MIN_REWARD_SITE_LEN

        print('Begin det. treadmill')

        patches = []
        for i in range(PATCH_TYPES_PER_ENV):
            def reward_func(site_idx):
                return 1
            patches.append(
                Patch(
                    n_reward_sites_for_patches[i],
                    reward_site_len_for_patches[i],
                    INTERREWARD_SITE_LEN_MEAN,
                    reward_func,
                    i,
                    reward_func_param=0.0,
                )
            )

        transition_mat = 1/3 * np.ones((PATCH_TYPES_PER_ENV, PATCH_TYPES_PER_ENV))

        sesh = TreadmillSession(
            patches,
            transition_mat,
            INTERPATCH_LEN,
            DWELL_TIME_FOR_REWARD,
            SPATIAL_BUFFER_FOR_VISUAL_CUES,
            obs_size=PATCH_TYPES_PER_ENV + 2,
            verbosity=False,
        )

        return sesh

    return make_env


def make_stochastic_treadmill_environment(env_idx):

    def make_env():
        n_reward_sites_for_patches = np.random.randint(MIN_N_REWARD_SITES_PER_PATCH, high=MAX_N_REWARD_SITES_PER_PATCH + 1, size=(PATCH_TYPES_PER_ENV,))
        reward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_SITE_LEN - MIN_REWARD_SITE_LEN) + MIN_REWARD_SITE_LEN
        decay_consts_for_reward_funcs = copy(REWARD_DECAY_CONSTS)
        np.random.shuffle(decay_consts_for_reward_funcs)

        print('Begin stoch. treadmill')
        print(decay_consts_for_reward_funcs)

        patches = []
        for i in range(PATCH_TYPES_PER_ENV):
            decay_const_for_i = decay_consts_for_reward_funcs[i]
            active = (decay_const_for_i != 0)
            def reward_func(site_idx, decay_const_for_i=decay_const_for_i, active=active):
                c = REWARD_PROB_PREFACTOR * np.exp(-site_idx / decay_const_for_i) if decay_const_for_i > 0 else 0
                if np.random.rand() < c and active:
                    return 1
                else:
                    return 0
            patches.append(
                Patch(
                    n_reward_sites_for_patches[i],
                    reward_site_len_for_patches[i],
                    INTERREWARD_SITE_LEN_MEAN,
                    reward_func,
                    i,
                    reward_func_param=(decay_consts_for_reward_funcs[i] if active else 0.0),
                )
            )

        transition_mat = 1/3 * np.ones((PATCH_TYPES_PER_ENV, PATCH_TYPES_PER_ENV))

        sesh = TreadmillSession(
            patches,
            transition_mat,
            INTERPATCH_LEN,
            DWELL_TIME_FOR_REWARD,
            SPATIAL_BUFFER_FOR_VISUAL_CUES,
            obs_size=PATCH_TYPES_PER_ENV + 2,
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
    hidden_state_output_dir = os.path.join(output_dir, 'hidden_state')
    make_path_if_not_exists(reward_rates_output_dir)
    make_path_if_not_exists(info_output_dir)
    make_path_if_not_exists(weights_output_dir)
    make_path_if_not_exists(hidden_state_output_dir)

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

    network.load_state_dict(torch.load(args.load_path, weights_only=True))
    network.eval()

    curricum = Curriculum(
        curriculum_step_starts=np.arange(8),
        curriculum_step_env_funcs=[
            make_stochastic_treadmill_environment,
        ] * 8,
    )

    env_seeds = np.arange(NUM_ENVS)
    save_num = 0
    last_snapshot = None

    with torch.no_grad():
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
            )

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
            
            hidden_states_for_session = agent.get_hidden_state_activities().cpu()

            padded_save_num = zero_pad(str(save_num), 5)
            np.save(os.path.join(reward_rates_output_dir, f'{padded_save_num}.npy'), avg_rewards_per_update)

            if session_num % OUTPUT_STATE_SAVE_RATE == 0 and session_num > 0:
                try:
                    compressed_write(all_info, os.path.join(info_output_dir, f'{padded_save_num}.pkl'))
                    torch.save(network.state_dict(), os.path.join(weights_output_dir, f'{padded_save_num}.h5'))
                    np.save(os.path.join(hidden_state_output_dir, f'{padded_save_num}.npy'), hidden_states_for_session)

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
    
    for var_noise in [0]:
        for activity_weight in [0]:
            print(objective(None, var_noise, activity_weight))