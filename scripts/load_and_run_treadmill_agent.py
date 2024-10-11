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
OBS_SIZE = PATCH_TYPES_PER_ENV + 2
ACTION_SIZE = 2
DWELL_TIME_FOR_REWARD = 6
SPATIAL_BUFFER_FOR_VISUAL_CUES = 1.5
MAX_REWARD_SITE_LEN = 2
MIN_REWARD_SITE_LEN = 2
MAX_N_REWARD_SITES_PER_PATCH = 16
MIN_N_REWARD_SITES_PER_PATCH = 16
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
ACTIVITY_WEIGHT = 1
VAR_NOISE = 0

# TRAINING PARAMS
NUM_ENVS = 20
N_UPDATES = 200
N_STEPS_PER_UPDATE = 200
N_UPDATES_PER_RESET = 25

# OTHER PARMS
DEVICE = 'cuda'
OUTPUT_SAVE_RATE = 200
OUTPUT_BASE_DIR = './data/rl_agent_outputs'

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str)
parser.add_argument('--load_path', metavar='lp', type=str)
args = parser.parse_args()



def make_stochastic_treadmill_environment(env_idx):

    def make_env():
        # np.random.seed(env_idx)
        
        n_reward_sites_for_patches = np.random.randint(MIN_N_REWARD_SITES_PER_PATCH, high=MAX_N_REWARD_SITES_PER_PATCH + 1, size=(PATCH_TYPES_PER_ENV,))
        reward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_SITE_LEN - MIN_REWARD_SITE_LEN) + MIN_REWARD_SITE_LEN
        decay_consts_for_reward_funcs = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_DECAY_CONST - MIN_REWARD_DECAY_CONST) + MIN_REWARD_DECAY_CONST
        inactive_patch = np.random.randint(0, high=PATCH_TYPES_PER_ENV)


        print('Begin stoch. treadmill')
        print(decay_consts_for_reward_funcs)

        patches = []
        for i in range(PATCH_TYPES_PER_ENV):
            decay_const_for_i = decay_consts_for_reward_funcs[i]
            active = (i != inactive_patch)
            def reward_func(site_idx, decay_const_for_i=decay_const_for_i, active=active):
                c = REWARD_PROB_PREFACTOR * np.exp(-site_idx / decay_const_for_i)
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
                    reward_func_param=(decay_consts_for_reward_funcs[i] if active else 0),
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
    hidden_activity_output_dir = os.path.join(output_dir, 'hidden_activity')
    make_path_if_not_exists(reward_rates_output_dir)
    make_path_if_not_exists(info_output_dir)
    make_path_if_not_exists(hidden_activity_output_dir)

    gamma = GAMMA
    learning_rate = LEARNING_RATE
    critic_weight = CRITIC_WEIGHT
    entropy_weight = ENTROPY_WEIGHT

    network = A2CRNN(
        input_size=OBS_SIZE + ACTION_SIZE + 1,
        action_size=ACTION_SIZE,
        hidden_size=HIDDEN_SIZE,
        device=DEVICE,
        var_noise=VAR_NOISE,
    )

    network.load_state_dict(torch.load(args.load_path, weights_only=True))
    network.eval()

    agent = A2CRecurrentAgent(
        network,
        action_space_dims=ACTION_SIZE,
        n_envs=NUM_ENVS,
        device=DEVICE,
        critic_weight=critic_weight, # changed for Optuna
        entropy_weight=entropy_weight, # changed for Optuna
        gamma=gamma, # changed for Optuna
        learning_rate=learning_rate, # changed for Optuna
        activity_weight=ACTIVITY_WEIGHT,
    )

    curricum = Curriculum(
        curriculum_step_starts=[t for t in range(0, 200, 25)],
        curriculum_step_env_funcs=[make_stochastic_treadmill_environment] * 8,
    )

    env_seeds = np.arange(NUM_ENVS)

    total_losses = np.empty((N_UPDATES))
    actor_losses = np.empty((N_UPDATES))
    critic_losses = np.empty((N_UPDATES))
    entropy_losses = np.empty((N_UPDATES))
    avg_rewards_per_update = np.empty((NUM_ENVS, N_UPDATES))
    all_hidden_states = np.empty((N_UPDATES * N_STEPS_PER_UPDATE, NUM_ENVS, HIDDEN_SIZE))
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

        hidden_states_for_update = agent.get_hidden_state_activities().detach().cpu()
        all_hidden_states[sample_phase * N_STEPS_PER_UPDATE: (sample_phase + 1) * N_STEPS_PER_UPDATE, ...] = hidden_states_for_update

        avg_rewards_per_update[:, sample_phase] = np.mean(total_rewards, axis=1)
        total_loss, actor_loss, critic_loss, entropy_loss = agent.get_losses()
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

        if sample_phase % N_UPDATES_PER_RESET == (N_UPDATES_PER_RESET - 1) and sample_phase > 0:
            print('sp', sample_phase)
            save_num = int(sample_phase / N_UPDATES_PER_RESET)
            padded_save_num = zero_pad(str(save_num), 5)
            np.save(os.path.join(reward_rates_output_dir, f'{padded_save_num}.npy'), avg_rewards_per_update[:, sample_phase - N_UPDATES_PER_RESET:sample_phase])
            pickle.dump(all_info, open(os.path.join(info_output_dir, f'{padded_save_num}.pkl'), 'wb'))
            np.save(os.path.join(hidden_activity_output_dir, f'{padded_save_num}.npy'), all_hidden_states[N_STEPS_PER_UPDATE * (sample_phase + 1 - N_UPDATES_PER_RESET):N_STEPS_PER_UPDATE * (sample_phase + 1)])


    final_value = avg_rewards_per_update[:, -1 * steps_before_prune:].mean()
    print('final_reward_total', final_value)

    return final_value

if __name__ == "__main__":
    print(objective(None))