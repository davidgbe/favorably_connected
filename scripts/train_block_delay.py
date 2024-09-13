if __name__ == '__main__':
    import sys
    from pathlib import Path
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import numpy as np
import torch
import os
import gymnasium as gym
from tqdm import tqdm
from environments.treadmill_session import TreadmillSession
from environments.components.patch import Patch
from environments.curriculum import Curriculum
from agents.networks.a2c_rnn import A2CRNN
from agents.a2c_recurrent_agent import A2CRecurrentAgent
from aux_funcs import zero_pad, make_path_if_not_exists
import optuna
from datetime import datetime
import argparse
import multiprocessing as mp
import pickle


# ENVIRONEMENT PARAMS
PATCH_TYPES_PER_ENV = 3
OBS_SIZE = 5
ACTION_SIZE = 2
DWELL_TIME_FOR_REWARD = 6
SPATIAL_BUFFER_FOR_VISUAL_CUES = 1.5
MAX_REWARD_SITE_LEN = 2
MIN_REWARD_SITE_LEN = 2
MAX_N_REWARD_SITES_PER_PATCH = 10
MIN_N_REWARD_SITES_PER_PATCH = 10
INTERREWARD_SITE_LEN_MEAN = 2
MAX_REWARD_DECAY_CONST = 30
MIN_REWARD_DECAY_CONST = 0.1
REWARD_PROB_PREFACTOR = 0.8
INTERPATCH_LEN = 6

# AGENT PARAMS
HIDDEN_SIZE = 128
CRITIC_WEIGHT = 0.07846647668470078
ENTROPY_WEIGHT = 1.0158892869509133e-06
GAMMA = 0.9867118269299845
LEARNING_RATE = 0.0006006712322528219

'''
{
    'gamma': 0.9867118269299845,
    'learning_rate': 0.0006006712322528219,
    'critic_weight': 0.07846647668470078,
    'entropy_weight': 1.0158892869509133e-06
}
'''

# TRAINING PARAMS
NUM_ENVS = 20
N_UPDATES = 20000
N_STEPS_PER_UPDATE = 200
N_UPDATES_PER_RESET = 25

# OTHER PARMS
DEVICE = 'cuda'
OUTPUT_SAVE_RATE = 200
OUTPUT_BASE_DIR = './data/rl_agent_block_curricula'

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str)
args = parser.parse_args()



def make_deterministic_treadmill_single_patch_environment(env_idx):

    def make_env():
        np.random.seed(env_idx)
        initial_patch_odor_num = np.random.randint(0, 3)
        
        print('Begin det. treadmill')

        patches = []

        def reward_func(site_idx):
            if site_idx < 3:
                return 1
            else:
                return 0

        patches.append(
            Patch(
                MIN_N_REWARD_SITES_PER_PATCH,
                MIN_REWARD_SITE_LEN,
                INTERREWARD_SITE_LEN_MEAN,
                reward_func,
                initial_patch_odor_num,
                reward_func_param=0,
            )
        )

        transition_mat = np.ones((1, 1))

        sesh = TreadmillSession(
            patches,
            transition_mat,
            INTERPATCH_LEN,
            DWELL_TIME_FOR_REWARD,
            SPATIAL_BUFFER_FOR_VISUAL_CUES,
            obs_size=OBS_SIZE,
            verbosity=False,
        )

        return sesh

    return make_env


def make_stochastic_two_treadmill_environment(env_idx):

    def make_env():
        np.random.seed(env_idx)
        initial_patch_odor_num = np.random.randint(0, 3)
        remaining_odor_nums = [0, 1, 2]
        remaining_odor_nums.remove(initial_patch_odor_num)
        second_odor_patch_num = remaining_odor_nums[np.random.randint(2)]
        
        patches = []

        def initial_patch_reward_func(site_idx):
            return 0

        patches.append(
            Patch(
                MIN_N_REWARD_SITES_PER_PATCH,
                MIN_REWARD_SITE_LEN,
                INTERREWARD_SITE_LEN_MEAN,
                initial_patch_reward_func,
                initial_patch_odor_num,
                reward_func_param=0,
            )
        )

        def second_patch_reward_func(site_idx):
            if site_idx < 3 and np.random.rand() < REWARD_PROB_PREFACTOR:
                return 1
            else:
                return 0

        patches.append(
            Patch(
                MIN_N_REWARD_SITES_PER_PATCH,
                MIN_REWARD_SITE_LEN,
                INTERREWARD_SITE_LEN_MEAN,
                second_patch_reward_func,
                second_odor_patch_num,
                reward_func_param=1,
            )
        )

        transition_mat = 1/2 * np.ones((2, 2))

        sesh = TreadmillSession(
            patches,
            transition_mat,
            INTERPATCH_LEN,
            DWELL_TIME_FOR_REWARD,
            SPATIAL_BUFFER_FOR_VISUAL_CUES,
            obs_size=OBS_SIZE,
            verbosity=False,
        )

        return sesh

    return make_env


def make_stochastic_three_treadmill_environment(env_idx):

    def make_env():
        np.random.seed(env_idx)
        initial_patch_odor_num = np.random.randint(0, 3)
        remaining_odor_nums = [0, 1, 2]
        remaining_odor_nums.remove(initial_patch_odor_num)
        second_odor_patch_num = remaining_odor_nums[np.random.randint(2)]
        remaining_odor_nums.remove(second_odor_patch_num)
        third_odor_patch_num = remaining_odor_nums[0]
        
        patches = []

        def initial_patch_reward_func(site_idx):
            return 0

        patches.append(
            Patch(
                MIN_N_REWARD_SITES_PER_PATCH,
                MIN_REWARD_SITE_LEN,
                INTERREWARD_SITE_LEN_MEAN,
                initial_patch_reward_func,
                initial_patch_odor_num,
                reward_func_param=0,
            )
        )

        def second_patch_reward_func(site_idx):
            if site_idx < 3 and np.random.rand() < REWARD_PROB_PREFACTOR:
                return 1
            else:
                return 0

        patches.append(
            Patch(
                MIN_N_REWARD_SITES_PER_PATCH,
                MIN_REWARD_SITE_LEN,
                INTERREWARD_SITE_LEN_MEAN,
                second_patch_reward_func,
                second_odor_patch_num,
                reward_func_param=1,
            )
        )

        def third_patch_reward_func(site_idx):
            if site_idx >= 3 and np.random.rand() < REWARD_PROB_PREFACTOR:
                return 1
            else:
                return 0

        patches.append(
            Patch(
                MIN_N_REWARD_SITES_PER_PATCH,
                MIN_REWARD_SITE_LEN,
                INTERREWARD_SITE_LEN_MEAN,
                third_patch_reward_func,
                third_odor_patch_num,
                reward_func_param=2,
            )
        )

        transition_mat = 1/3 * np.ones((3, 3))

        sesh = TreadmillSession(
            patches,
            transition_mat,
            INTERPATCH_LEN,
            DWELL_TIME_FOR_REWARD,
            SPATIAL_BUFFER_FOR_VISUAL_CUES,
            obs_size=OBS_SIZE,
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
    )

    agent = A2CRecurrentAgent(
        network,
        action_space_dims=ACTION_SIZE,
        n_envs=NUM_ENVS,
        device=DEVICE,
        critic_weight=critic_weight, # changed for Optuna
        entropy_weight=entropy_weight, # changed for Optuna
        gamma=gamma, # changed for Optuna
        learning_rate=learning_rate, # changed for Optuna
    )

    curricum = Curriculum(
        curriculum_step_starts=[0, 1000, 5000],
        curriculum_step_env_funcs=[
            make_deterministic_treadmill_single_patch_environment,
            make_stochastic_two_treadmill_environment,
            make_stochastic_three_treadmill_environment,
        ],
    )

    env_seeds = np.arange(NUM_ENVS)

    total_losses = np.empty((N_UPDATES))
    actor_losses = np.empty((N_UPDATES))
    critic_losses = np.empty((N_UPDATES))
    entropy_losses = np.empty((N_UPDATES))
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
            action = action.cpu().numpy()
            obs, reward, terminated, truncated, info = envs.step(action)
            agent.append_reward(reward.astype('float32'))
            total_rewards[:, step] = reward
            all_info.append(info)

        avg_rewards_per_update[:, sample_phase] = np.mean(total_rewards, axis=1)
        total_loss, actor_loss, critic_loss, entropy_loss = agent.get_losses()
        agent.update(total_loss)
        reset_network = sample_phase % N_UPDATES_PER_RESET == 0 and sample_phase > 0
        if reset_network:
            agent.reset_state()
        else:
            hidden_states = agent.reset_state()
            agent.set_state(hidden_states)

        total_losses[sample_phase] = total_loss.detach().cpu().numpy()
        actor_losses[sample_phase] = actor_loss.detach().cpu().numpy()
        critic_losses[sample_phase] = critic_loss.detach().cpu().numpy()
        entropy_losses[sample_phase] = entropy_loss.detach().cpu().numpy()

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
    print('final_reward_total', final_value)
    print('gamma', gamma)
    print('learning_rate', learning_rate)
    print('critic_weight', critic_weight)
    print('entropy_weight', entropy_weight)

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