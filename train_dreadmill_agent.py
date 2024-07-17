import numpy as np
import torch
import os
import gymnasium as gym
from tqdm import tqdm
from environments.dreadmill_session import DreadmillSession
from environments.components.patch import Patch
from agents.networks.a2c_rnn import A2CRNN
from agents.a2c_recurrent_agent import A2CRecurrentAgent

# ENVIRONEMENT PARAMS
PATCH_TYPES_PER_ENV = 3
OBS_SIZE = PATCH_TYPES_PER_ENV + 2
ACTION_SIZE = 2
DWELL_TIME_FOR_REWARD = 6
SPATIAL_BUFFER_FOR_VISUAL_CUES = 1.5
MAX_REWARD_SITE_LEN = 5
MIN_REWARD_SITE_LEN = 2
MAX_INTERREWARD_SITE_LEN = 3
MIN_INTERREWARD_SITE_LEN = 1
MAX_REWARD_DECAY_CONST = 10
MIN_REWARD_DECAY_CONST = 0.1
REWARD_PROB_PREFACTOR = 0.8

# AGENT PARAMS
HIDDEN_SIZE = 256
CRITIC_WEIGHT = 0.05
ENTROPY_WEIGHT = 0.05
GAMMA = 0
LEARNING_RATE = 0.001

# TRAINING PARAMS
NUM_ENVS=12
N_UPDATES = 500000
N_STEPS_PER_UPDATE = 15

DEVICE = 'cuda'
OUTPUT_DIR = ''


def make_dreadmill_environment(i):

    def make_env():
        
        n_reward_sites_for_patches = np.random.randint(2, high=4, size=(PATCH_TYPES_PER_ENV,))
        reward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_SITE_LEN - MIN_REWARD_SITE_LEN) + MIN_REWARD_SITE_LEN
        interreward_site_len_for_patches = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_INTERREWARD_SITE_LEN - MIN_INTERREWARD_SITE_LEN) + MIN_INTERREWARD_SITE_LEN
        decay_consts_for_reward_funcs = np.random.rand(PATCH_TYPES_PER_ENV) * (MAX_REWARD_DECAY_CONST - MIN_REWARD_DECAY_CONST) + MIN_REWARD_DECAY_CONST

        patches = []
        for i in range(PATCH_TYPES_PER_ENV):
            def reward_func(site_idx):
                c = REWARD_PROB_PREFACTOR * np.exp(-site_idx / decay_consts_for_reward_funcs[i])
                if np.random.rand() < c:
                    return 1
                else:
                    return 0
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

def main():
    envs = gym.vector.AsyncVectorEnv([make_dreadmill_environment(i) for i in range(NUM_ENVS)])

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
        critic_weight=CRITIC_WEIGHT, 
        entropy_weight=ENTROPY_WEIGHT, 
        gamma=GAMMA, 
        learning_rate=LEARNING_RATE
    )

    total_losses = np.empty((N_UPDATES))
    actor_losses = np.empty((N_UPDATES))
    critic_losses = np.empty((N_UPDATES))
    entropy_losses = np.empty((N_UPDATES))
    avg_rewards_per_update = np.empty((NUM_ENVS, N_UPDATES))
    for sample_phase in tqdm(range(N_UPDATES)):
        # at the start of training reset all envs to get an initial state
        # play n steps in our parallel environments to collect data
        obs, info = envs.reset()
        print('gets here')
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

if __name__ == "__main__":
    main()