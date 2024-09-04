if __name__ == '__main__':
    import sys
    from pathlib import Path
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import numpy as np
import os
import gymnasium as gym
from tqdm import tqdm
from environments.treadmill_session import TreadmillSession
from environments.components.patch import Patch
from environments.curriculum import Curriculum
from agents.grid_search_agent import GridSearchAgent
from aux_funcs import zero_pad, make_path_if_not_exists
import optuna
from datetime import datetime
import argparse
import multiprocessing as mp
import pickle


# ENVIRONEMENT PARAMS
PATCH_TYPES_PER_ENV = 3
OBS_SIZE = PATCH_TYPES_PER_ENV + 2
ACTION_SIZE = 2
DWELL_TIME_FOR_REWARD = 6
SPATIAL_BUFFER_FOR_VISUAL_CUES = 1.5
MAX_REWARD_SITE_LEN = 2
MIN_REWARD_SITE_LEN = 2
MAX_N_REWARD_SITES_PER_PATCH = 8
MIN_N_REWARD_SITES_PER_PATCH = 8
INTERREWARD_SITE_LEN_MEAN = 2
MAX_REWARD_DECAY_CONST = 30
MIN_REWARD_DECAY_CONST = 0.1
REWARD_PROB_PREFACTOR = 0.8
INTERPATCH_LEN = 6


# TRAINING PARAMS
NUM_ENVS = 20
N_UPDATES = 20000
N_STEPS_PER_UPDATE = 200
N_UPDATES_PER_RESET = 25

# OTHER PARMS
OUTPUT_SAVE_RATE = 200
OUTPUT_BASE_DIR = './data/grid_search_agent_outputs'

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str)
args = parser.parse_args()



def make_expected_treadmill_environment(env_idx):

    def make_env():
        np.random.seed(env_idx + 2)
        
        n_reward_sites_for_patches = np.random.randint(MIN_N_REWARD_SITES_PER_PATCH, high=MAX_N_REWARD_SITES_PER_PATCH + 1, size=(PATCH_TYPES_PER_ENV,))
        reward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_SITE_LEN - MIN_REWARD_SITE_LEN) + MIN_REWARD_SITE_LEN
        decay_consts_for_reward_funcs = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_DECAY_CONST - MIN_REWARD_DECAY_CONST) + MIN_REWARD_DECAY_CONST

        print('Begin expected. treadmill')
        print(decay_consts_for_reward_funcs)

        patches = []
        for i in range(PATCH_TYPES_PER_ENV):
            decay_const_for_i = decay_consts_for_reward_funcs[i]
            def reward_func(site_idx, decay_const_for_i=decay_const_for_i):
                c = REWARD_PROB_PREFACTOR * np.exp(-site_idx / decay_const_for_i)
                return c
            patches.append(
                Patch(
                    n_reward_sites_for_patches[i],
                    reward_site_len_for_patches[i],
                    INTERREWARD_SITE_LEN_MEAN,
                    reward_func,
                    i,
                    reward_func_param=decay_consts_for_reward_funcs[i],
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
        np.random.seed(env_idx + 2)
        
        n_reward_sites_for_patches = np.random.randint(MIN_N_REWARD_SITES_PER_PATCH, high=MAX_N_REWARD_SITES_PER_PATCH + 1, size=(PATCH_TYPES_PER_ENV,))
        reward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_SITE_LEN - MIN_REWARD_SITE_LEN) + MIN_REWARD_SITE_LEN
        decay_consts_for_reward_funcs = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_DECAY_CONST - MIN_REWARD_DECAY_CONST) + MIN_REWARD_DECAY_CONST

        print('Begin stoch. treadmill')
        print(decay_consts_for_reward_funcs)

        patches = []
        for i in range(PATCH_TYPES_PER_ENV):
            decay_const_for_i = decay_consts_for_reward_funcs[i]
            def reward_func(site_idx, decay_const_for_i=decay_const_for_i):
                c = REWARD_PROB_PREFACTOR * np.exp(-site_idx / decay_const_for_i)
                if np.random.rand() < c:
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
                    reward_func_param=decay_consts_for_reward_funcs[i],
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



def objective(trial):
    time_stamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    output_dir = os.path.join(OUTPUT_BASE_DIR, '_'.join([args.exp_title, time_stamp]))
    reward_rates_output_dir = os.path.join(output_dir, 'reward_rates')
    info_output_dir = os.path.join(output_dir, 'state')
    make_path_if_not_exists(reward_rates_output_dir)
    make_path_if_not_exists(info_output_dir)


    agent = GridSearchAgent(
        n_envs=NUM_ENVS,
        wait_time_for_reward=DWELL_TIME_FOR_REWARD - MIN_REWARD_SITE_LEN,
        odor_cues_indices=(2, 5),
        patch_cue_idx=0,
        max_stops_per_patch=8,
    )

    curricum = Curriculum(
        curriculum_step_starts=[0, 18500],
        curriculum_step_env_funcs=[
            make_expected_treadmill_environment,
            make_stochastic_treadmill_environment,
        ],
    )

    env_seeds = np.arange(NUM_ENVS)

    avg_rewards_per_update = np.empty((NUM_ENVS, N_UPDATES))
    all_info = []

    for sample_phase in tqdm(range(N_UPDATES)):
        # at the start of training reset all envs to get an initial state
        # play n steps in our parallel environments to collect data
        envs = curricum.get_envs_for_step(env_seeds)
        if sample_phase % N_UPDATES_PER_RESET == 0:
            obs, info = envs.reset()
            all_info = []

        total_rewards = np.empty((NUM_ENVS, N_STEPS_PER_UPDATE))
        for step in range(N_STEPS_PER_UPDATE):
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = envs.step(action)
            agent.append_reward(reward.astype('float32'))
            total_rewards[:, step] = reward
            all_info.append(info)

        avg_rewards_per_update[:, sample_phase] = np.mean(total_rewards, axis=1)
        reset_network = sample_phase % N_UPDATES_PER_RESET == 0 and sample_phase > 0
        if reset_network:
            agent.cue_session_end()
            agent.reset_state()

        steps_before_prune = 100

        if trial is not None and sample_phase > 0:
            if sample_phase > steps_before_prune:
                intermediate_value = avg_rewards_per_update[:, sample_phase - steps_before_prune : sample_phase].mean()
            else:
                intermediate_value = avg_rewards_per_update[:, :sample_phase].mean()
            
            trial.report(intermediate_value, sample_phase)

            if trial.should_prune():
                raise optuna.TrialPruned()

        if sample_phase % OUTPUT_SAVE_RATE == OUTPUT_SAVE_RATE - 1:
            save_num = int(sample_phase / OUTPUT_SAVE_RATE)
            padded_save_num = zero_pad(str(save_num), 5)
            np.save(os.path.join(reward_rates_output_dir, f'{padded_save_num}.npy'), avg_rewards_per_update[:, sample_phase - OUTPUT_SAVE_RATE + 1:sample_phase])
            pickle.dump(all_info, open(os.path.join(info_output_dir, f'{padded_save_num}.pkl'), 'wb'))


    final_value = avg_rewards_per_update[:, -1 * steps_before_prune:].mean()

    return final_value

if __name__ == "__main__":
    
    # study = optuna.create_study(
    #     direction='maximize',
    #     pruner=optuna.pruners.MedianPruner(
    #         n_startup_trials=5,
    #         n_warmup_steps=5000,
    #         interval_steps=10,
    #     ),
    # )
    # study.optimize(objective, n_trials=20)
    # print(study.best_params)

    print(objective(None))