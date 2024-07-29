import numpy as np
import os
import gymnasium as gym
from tqdm import tqdm
from environments.dreadmill_session import DreadmillSession
from environments.components.patch import Patch
from aux_funcs import zero_pad
from agents.naive_agent import NaiveAgent

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

# TRAINING PARAMS
NUM_ENVS=12
N_UPDATES = 1000
N_STEPS_PER_UPDATE = 200

# OTHER PARMS
DEVICE = 'cuda'
OUTPUT_SAVE_RATE = 200
OUTPUT_DIR = './data/naive_agent_outputs'


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

def main():
    envs = gym.vector.AsyncVectorEnv([make_dreadmill_environment(i) for i in range(NUM_ENVS)])

    agent = NaiveAgent(
        n_envs=NUM_ENVS,
        wait_time_for_reward=DWELL_TIME_FOR_REWARD,
        odor_cues_indicies=(2, 2 + PATCH_TYPES_PER_ENV),
    )

    avg_rewards_per_update = np.empty((NUM_ENVS, N_UPDATES))

    for sample_phase in tqdm(range(N_UPDATES)):
        # at the start of training reset all envs to get an initial state
        # play n steps in our parallel environments to collect data
        obs, info = envs.reset()

        total_rewards = np.empty((NUM_ENVS, N_STEPS_PER_UPDATE))
        for step in range(N_STEPS_PER_UPDATE):
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = envs.step(action)
            agent.append_reward(reward.astype('float32'))
            total_rewards[:, step] = reward
        avg_rewards_per_update[:, sample_phase] = np.mean(total_rewards, axis=1)

        steps_before_prune = 100

        # print('actor', actor_losses[sample_phase])
        # print('critic', CRITIC_WEIGHT * critic_losses[sample_phase])
        # print('entropy', ENTROPY_WEIGHT * entropy_losses[sample_phase])

        if sample_phase % OUTPUT_SAVE_RATE == 0 and sample_phase > 0:
            save_num = int(sample_phase / OUTPUT_SAVE_RATE)
            padded_save_num = zero_pad(str(save_num), 5)
            np.save(os.path.join(OUTPUT_DIR, f'mean_rewards_per_update_{padded_save_num}.npy'), avg_rewards_per_update[:, sample_phase - OUTPUT_SAVE_RATE:sample_phase])
    print(avg_rewards_per_update)
    final_value = avg_rewards_per_update[:, -1 * steps_before_prune:].mean()
    print('final_reward_total', final_value)

    return final_value

if __name__ == "__main__":
    print(main())