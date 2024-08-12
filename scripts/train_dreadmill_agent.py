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
from environments.dreadmill_session import DreadmillSession
from environments.components.patch import Patch
from agents.networks.a2c_rnn import A2CRNN
from agents.a2c_recurrent_agent import A2CRecurrentAgent
from aux_funcs import zero_pad, make_path_if_not_exists
import optuna
from datetime import datetime
import argparse
import multiprocessing as mp



# ENVIRONEMENT PARAMS
PATCH_TYPES_PER_ENV = 3
OBS_SIZE = PATCH_TYPES_PER_ENV + 2
ACTION_SIZE = 2
DWELL_TIME_FOR_REWARD = 6
SPATIAL_BUFFER_FOR_VISUAL_CUES = 1.5
MAX_REWARD_SITE_LEN = 5
MIN_REWARD_SITE_LEN = 2
MAX_N_REWARD_SITES_PER_PATCH = 4
MIN_N_REWARD_SITES_PER_PATCH = 2
MAX_INTERREWARD_SITE_LEN = 3
MIN_INTERREWARD_SITE_LEN = 1
MAX_REWARD_DECAY_CONST = 10
MIN_REWARD_DECAY_CONST = 0.1
REWARD_PROB_PREFACTOR = 0.8

# AGENT PARAMS
HIDDEN_SIZE = 32
CRITIC_WEIGHT = 0.008462461633690228
ENTROPY_WEIGHT = 9.105961307700953e-05
GAMMA = 0.776958910455669
LEARNING_RATE = 0.0018679247001861746

 # {'gamma': 0.7769589104556699, 'learning_rate': 0.0018679247001861746, 'critic_weight': 0.008462461633690228, 'entropy_weight': 9.105961307700953e-05}
 # final value:  0.07893333333333333

# TRAINING PARAMS
NUM_ENVS=12
N_UPDATES = 20000
N_STEPS_PER_UPDATE = 200

# OTHER PARMS
DEVICE = 'cuda'
OUTPUT_SAVE_RATE = 20
OUTPUT_BASE_DIR = './data/rl_agent_outputs'

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str)
args = parser.parse_args()


def make_dreadmill_environment(env_idx):

    def make_env():
        np.random.seed(env_idx)
        
        n_reward_sites_for_patches = np.random.randint(MIN_N_REWARD_SITES_PER_PATCH, high=MAX_N_REWARD_SITES_PER_PATCH + 1, size=(PATCH_TYPES_PER_ENV,))
        reward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_SITE_LEN - MIN_REWARD_SITE_LEN) + MIN_REWARD_SITE_LEN
        interreward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_INTERREWARD_SITE_LEN - MIN_INTERREWARD_SITE_LEN) + MIN_INTERREWARD_SITE_LEN
        decay_consts_for_reward_funcs = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_DECAY_CONST - MIN_REWARD_DECAY_CONST) + MIN_REWARD_DECAY_CONST

        patches = []
        for i in range(PATCH_TYPES_PER_ENV):
            def reward_func(site_idx):
                # c = REWARD_PROB_PREFACTOR * np.exp(-site_idx / decay_consts_for_reward_funcs[i])
                # if np.random.rand() < c:
                #     return 1
                # else:
                #     return 0
                return 1
            patches.append(Patch(n_reward_sites_for_patches[i], reward_site_len_for_patches[i], interreward_site_len_for_patches[i], reward_func, i))

        transition_mat = 1/3 * np.ones((PATCH_TYPES_PER_ENV, PATCH_TYPES_PER_ENV))

        sesh = DreadmillSession(
            patches,
            transition_mat,
            10,
            DWELL_TIME_FOR_REWARD,
            SPATIAL_BUFFER_FOR_VISUAL_CUES,
            verbosity=False,
        )

        return sesh

    return make_env

def objective(trial):

    if trial is not None:
        gamma = trial.suggest_float('gamma', 0.0, 1.0)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 0.01, log=True)
        critic_weight = trial.suggest_float('critic_weight', 1e-6, 0.1, log=True)
        entropy_weight = trial.suggest_float('entropy_weight', 1e-6, 0.1, log=True)
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

    envs = gym.vector.AsyncVectorEnv([make_dreadmill_environment(i) for i in range(NUM_ENVS)])

    total_losses = np.empty((N_UPDATES))
    actor_losses = np.empty((N_UPDATES))
    critic_losses = np.empty((N_UPDATES))
    entropy_losses = np.empty((N_UPDATES))
    avg_rewards_per_update = np.empty((NUM_ENVS, N_UPDATES))

    for sample_phase in tqdm(range(N_UPDATES)):
        # at the start of training reset all envs to get an initial state
        # play n steps in our parallel environments to collect data
        obs, info = envs.reset()

        total_rewards = np.empty((NUM_ENVS, N_STEPS_PER_UPDATE))
        for step in range(N_STEPS_PER_UPDATE):
            action = agent.sample_action(obs)
            action = action.cpu().numpy()
            obs, reward, terminated, truncated, info = envs.step(action)
            agent.append_reward(reward.astype('float32'))
            total_rewards[:, step] = reward
        avg_rewards_per_update[:, sample_phase] = np.mean(total_rewards, axis=1)
        total_loss, actor_loss, critic_loss, entropy_loss = agent.get_losses()
        agent.update(total_loss)
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
        else:
            # print('actor', actor_losses[sample_phase])
            # print('critic', CRITIC_WEIGHT * critic_losses[sample_phase])
            # print('entropy', ENTROPY_WEIGHT * entropy_losses[sample_phase])
            if sample_phase % OUTPUT_SAVE_RATE == 0 and sample_phase > 0:
                save_num = int(sample_phase / OUTPUT_SAVE_RATE)
                padded_save_num = zero_pad(str(save_num), 5)
                np.save(os.path.join(output_dir, f'mean_rewards_per_update_{padded_save_num}.npy'), avg_rewards_per_update[:, sample_phase - OUTPUT_SAVE_RATE:sample_phase])
    
    final_value = avg_rewards_per_update[:, -1 * steps_before_prune:].mean()
    print('final_reward_total', final_value)
    print('gamma', gamma)
    print('learning_rate', learning_rate)
    print('critic_weight', critic_weight)
    print('entropy_weight', entropy_weight)

    return final_value

if __name__ == "__main__":
    time_stamp = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    output_dir = os.path.join(OUTPUT_BASE_DIR, '_'.join([args.exp_title, time_stamp]))
    make_path_if_not_exists(output_dir)
    
    # study = optuna.create_study(
    #     direction='maximize',
    #     pruner=optuna.pruners.MedianPruner(
    #         n_startup_trials=5,
    #         n_warmup_steps=100,
    #         interval_steps=10,
    #     ),
    # )
    # study.optimize(objective, n_trials=100)
    # print(study.best_params)

    print(objective(None))